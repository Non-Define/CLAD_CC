# Define Model (XLSR-ORIG)
# by HH
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Callable, Optional, Union
from transformers import Wav2Vec2Model

'''
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
print(model.config)
'''

class ConvLayers(nn.Module):
    def __init__(self, input_dim=1024, proj_dim=256, conv_channels=32):
        super(ConvLayers, self).__init__()

        # 1. Conv1D projection: (B, T, 1024) → (B, T, 256)
        self.proj = nn.Conv1d(in_channels=input_dim,
                              out_channels=proj_dim,
                              kernel_size=1)

        # 2. Conv2D blocks: apply 3 times
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B, T, 1024)
        x = x.transpose(1, 2)          # (B, 1024, T)
        x = self.proj(x)               # (B, 256, T)
        x = x.transpose(1, 2)          # (B, T, 256)

        # Conv2D expects 4D: (B, C, H, W), so add channel dim
        x = x.unsqueeze(1)             # (B, 1, T, 256)
        x = self.convs(x)              # (B, 32, T, 256)
        x = x.permute(0, 2, 3, 1)      # (B, T, 256, 32) 
        return x
    
class SELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """
    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(SELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class SERe2blocks(nn.Module):
    def __init__(self, input_dim=32, conv_channels=32, scale=4, reduction_ratio=2):
        super().__init__()
        assert conv_channels % scale == 0
        self.scale = scale
        self.width = conv_channels // scale

        self.conv1 = nn.Conv2d(input_dim, conv_channels, kernel_size=1, bias=False)  
        self.bn1   = nn.BatchNorm2d(conv_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.width, self.width, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.width),
                nn.ReLU(inplace=True)
            ) for _ in range(scale - 1)
        ])
        self.bn3   = nn.BatchNorm2d(input_dim)
        self.se    = SELayer(num_channels=input_dim, reduction_ratio=reduction_ratio)

    def forward(self, x):
        x = x.permute(0,3,1,2)
        out = self.conv1(x)
        out = self.bn1(out)

        spl = torch.chunk(out, self.scale, dim=1)
        y = [spl[0]]
        for i in range(1, self.scale):
            yi = self.conv3[i-1](spl[i] + y[i-1])
            y.append(yi)

        out = torch.cat(y, dim=1)
        out = self.conv1(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.se(out)
        out = out.permute(0,2,3,1)    # (B, T, 256, 32)
        return out
    
class 
#---------------------------------------------------------------------------------------
# Model
class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = ConvLayers()  # (B, 1, 201, 256) → (B, 32, 201, 256)

        def pool_block(pool_kernel, pool_stride):
            return nn.Sequential(
                Permute(0, 3, 1, 2),  # (B, T, F, C) → (B, C, T, F)
                nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride),
                Permute(0, 2, 3, 1),  # (B, C, T', F') → (B, T', F', C)
            )
        self.encoder = nn.Sequential(
            SERe2blocks(input_dim=32),  # Block 1
            nn.Sequential(             # Block 2: (201,256) → (100,128)
                pool_block(pool_kernel=2, pool_stride=2),
                SERe2blocks(input_dim=32)
            ),
            SERe2blocks(input_dim=32),  # Block 3
            nn.Sequential(             # Block 4: (100,128) → (50,64)
                pool_block(pool_kernel=2, pool_stride=2),
                SERe2blocks(input_dim=32)
            ),
            SERe2blocks(input_dim=32),  # Block 5
            nn.Sequential(             # Block 6: (50,64) → (25,32)
                pool_block(pool_kernel=2, pool_stride=2),
                SERe2blocks(input_dim=32)
            ),
            SERe2blocks(input_dim=32),  # Block 7
            nn.Sequential(             # Block 8: (25,32) → (25,16)
                pool_block(pool_kernel=(1, 2), pool_stride=(1, 2)),
                SERe2blocks(input_dim=32)
            ),
        )

    def forward(self, x):
        x = self.frontend(x)  # (B, 201, 256, 32)
        x = self.encoder(x)   # (B, 25, 16, 32)
        return x