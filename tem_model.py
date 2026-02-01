import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.autograd import Variable
import math

# --------------------------------------------------------------------------------------------------
# SincNet
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def sinc(band,t_right):
    y_right= torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
    y_left= flip(y_right,0)

    y=torch.cat([y_left,Variable(torch.ones(1)).cuda(),y_right])

    return y
    
class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels=15, kernel_size=251, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast,self).__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)
        
        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);


        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate/2)
        band = (high - low)[:, 0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])
    
        filters = (band_pass).view(self.out_channels, 1, self.kernel_size)

        low_filters = filters[0:10, :, :]
        low_freq = F.conv1d(waveforms, low_filters, stride=self.stride,
                            padding=self.padding, dilation=self.dilation,
                            bias=None, groups=1) # (B, 10, 64350)

        high_filters = filters[10:15, :, :]
        high_freq = F.conv1d(waveforms, high_filters, stride=self.stride,
                             padding=self.padding, dilation=self.dilation,
                             bias=None, groups=1) # (B, 5, 64350)

        return low_freq, high_freq
# --------------------------------------------------------------------------------------------------
# Gating-Res2Net
class GatingRe2blocks(nn.Module):
    def __init__(self, out_channels=125, scale=5):
        super(GatingRe2blocks, self).__init__()
        self.scale = scale
        self.width = out_channels // scale
        
        self.conv1_list = nn.ModuleList([
            nn.Conv1d(2, self.width, kernel_size=1) for _ in range(scale)
        ])
        self.conv3_list = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size=3, padding=1) for _ in range(scale - 1)
        ])
        self.gating_network = nn.Sequential(
            nn.Linear(out_channels, out_channels // 4), 
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 4, scale),        
            nn.Sigmoid()                                 
        )
        self.identity_proj = nn.Conv1d(10, out_channels, kernel_size=1)
        self.proj_final = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn_list = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(scale)])
        self.relu = nn.ReLU(inplace=True)
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.gap = nn.AdaptiveAvgPool1d(1)              
        self.ln = nn.LayerNorm(out_channels)         
        self.fc1 = nn.Linear(out_channels, 64)          # 125 -> 64
        self.leaky_relu = nn.LeakyReLU(0.2)             
        self.dropout = nn.Dropout(p=0.3)                
        self.fc2 = nn.Linear(64, 2)                     # 64 -> 2

    def forward(self, low_freq):
        identity = self.identity_proj(low_freq)
        low_grouped = torch.chunk(low_freq, self.scale, dim=1)
        
        y = [None] * self.scale
        y[4] = self.relu(self.bn_list[4](self.conv1_list[4](low_grouped[4])))
        
        for i in range(self.scale - 2, -1, -1):
            x_i = self.conv1_list[i](low_grouped[i])
            yi = self.conv3_list[i](x_i + y[i+1])
            y[i] = self.relu(self.bn_list[i](yi))
            
        ms_feat = torch.cat(y, dim=1)
        avg_pool = torch.mean(ms_feat, dim=-1)
        gate_weights = self.gating_network(avg_pool) 
 
        y_weighted = []
        for i in range(self.scale):
            w = gate_weights[:, i].view(-1, 1, 1)
            y_weighted.append(y[i] * w)
        
        ms_feat_gated = torch.cat(y_weighted, dim=1)
        ms_feat_gated = self.proj_final(ms_feat_gated)
        
        out = self.relu(self.final_bn(ms_feat_gated + identity))  # (B, 125, 64350)
        out_pooled = self.gap(out).squeeze(-1)          # (B, 125, 1) -> (B, 125)
        out_normed = self.ln(out_pooled)                
        
        x = self.fc1(out_normed)
        x = self.leaky_relu(x)                          
        x = self.dropout(x)
        final_output = self.fc2(x)                      # (B, 2)
        
        return final_output
