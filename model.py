import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Union

# the over all MoCo framework to train the encoder   
# mlp is the projection head, if mlp is True, then the projection head will be used, otherwise, the projection head will not be used 
# Normally queue_feature_dim is set as the feature dim output by encoder
class MoCo_v2(nn.Module):
    def __init__(self, encoder_q, encoder_k, queue_feature_dim, queue_size=65536, momentum=0.999, temperature=0.07, mlp=False, return_q = False):
        super(MoCo_v2, self).__init__()
        self.return_q = return_q
        # Initialize the momentum coefficient and temperature parameter
        self.momentum = momentum
        self.temperature = temperature
        self.queue_size = queue_size
        # self.weight_decay = 1e-4  # follow the official implementation
        self.mlp = mlp
        self.projection_dim = 128
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        # make sure that two encoders have the same params
        self.encoder_k.load_state_dict(self.encoder_q.state_dict())
        self.queue_feature_dim = queue_feature_dim
         
        if mlp: # if projection head is used
            self.projection_head_q = nn.Sequential(
                nn.Linear(queue_feature_dim, queue_feature_dim),
                nn.ReLU(),
                nn.Linear(queue_feature_dim, self.projection_dim)
            )
            self.projection_head_k = nn.Sequential(
                nn.Linear(queue_feature_dim, queue_feature_dim),
                nn.ReLU(),
                nn.Linear(queue_feature_dim, self.projection_dim)
            )
            # make sure that two encoders have the same params
            self.projection_head_k.load_state_dict(self.projection_head_q.state_dict())
            self.queue_feature_dim = self.projection_dim
        # Initialize the queue
        self.queue = torch.randn(self.queue_size, self.queue_feature_dim)
        self.queue = F.normalize(self.queue, dim=1)  # the official implementation also normalize the queue, actully they make q and k always a unit vector, so that dot product gives the cosine similarity
        self.queue_ptr = 0   

    def init_queue(self):
        # Initialize the queue, and make sure the queue is always on the same device with MoCo Module.
        device = next(self.parameters()).device
        self.queue = torch.randn(self.queue_size, self.queue_feature_dim, device=device)
        self.queue = F.normalize(self.queue, dim=1)  # the official implementation also normalize the queue, actully they make q and k always a unit vector, so that dot product gives the cosine similarity
        self.queue_ptr = 0    
    
    # overriding the to function, so that the queue can be moved to the GPU
    def to(self, *args, **kwargs):
        self = super(MoCo_v2, self).to(*args, **kwargs)
        self.queue = self.queue.to(*args, **kwargs)
        return self

    @torch.no_grad()    
    def momentum_update(self):
        """
        Update the key encoder with the momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
        if self.mlp:
            for param_q, param_k in zip(self.projection_head_q.parameters(), self.projection_head_k.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()        
    def dequeue_and_enqueue(self, keys):
        """
        Dequeue the oldest keys and enqueue the new keys into the queue
        """
        batch_size = keys.shape[0]
        assert self.queue_size % batch_size == 0  # for simplicity
        
        # Enqueue the keys into the queue
        self.queue[self.queue_ptr:self.queue_ptr+batch_size] = keys
        self.queue_ptr = (self.queue_ptr + batch_size) % self.queue.shape[0]
        
    def forward(self, x_q, x_k):
        """
        Input:
            x_q: a batch of query audio
            x_k: a batch of key audio
        Output:
            logits, targets, feature_q(optional)
        """
        # At here we follow the official implementation of MoCo v2, https://github.com/facebookresearch/moco/blob/main/moco/builder.py
        # Encode the input batch with the query encoder
        q = self.encoder_q(x_q) # queries: NxC
        
        if self.mlp:
            q = self.projection_head_q(q)
        if self.return_q:
            q_without_normal = q
        q = nn.functional.normalize(q, dim=1) 
        # compute key features
        with torch.no_grad():
            # Update the momentum encoder with the current encoder
            self.momentum_update()            
            # Encode the input batch with the key encoder
            k = self.encoder_k(x_k)
            if self.mlp:
                k = self.projection_head_k(k)
            k = nn.functional.normalize(k, dim=1)
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach().T])  # here we modified the official implementation, we transpose the queue, since it is KxC, and we want to do dot product with NxC, so we need to transpose the queue  
            
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        if torch.cuda.is_available():
            labels = labels.cuda()

        # Dequeue the oldest keys and enqueue the new keys
        self.dequeue_and_enqueue(k)
        
        if self.return_q:
            return logits, labels, q_without_normal
        else:
            return logits, labels

# # print how many parameters the New Transformer Decoder Layer has
# total_params = sum(
#     param.numel() for param in TransformerDecoderAggregatorLayer(d_model=last_hidden_state.shape[-1], nhead=8, batch_first=True).parameters()
# )
# print(total_params)

# # print how many parameters the Original Transformer Decoder has
# total_params = sum(
#     param.numel() for param in torch.nn.TransformerDecoderLayer(d_model=last_hidden_state.shape[-1], nhead=8, batch_first=True).parameters()
# )
# print(total_params)
# define the encoder which is just a MLP

class ContrastiveLearningEncoderMLP(nn.Module):
    # init a wav2vec2 model
    def __init__(self, wav2vec2_path):
        super(ContrastiveLearningEncoderMLP, self).__init__()

        self.fc1 = nn.Linear(64600, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1024)

        self.feature_dim = 1024  # the final out put will be 1024 dims.

    # define the forward function, the input x should be a batch of audio files, has shape (batch_size, 64600), and the output should have shape (batch_size, 1024)    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# define the downstream classifier used for spoof detection with only 1 linear layer without activation
class DownStreamLinearClassifier(nn.Module):
    def __init__(self, encoder, input_depth=1024):
        super(DownStreamLinearClassifier, self).__init__()
        self.input_depth = input_depth
        self.encoder = encoder  # this should be able to encoder the input(batch_size, 64600) into feature vectors(batch_size, input_depth)
        self.fc = nn.Linear(input_depth, 2)  

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        x = x.squeeze(1)
        return x

# define the downstream classifier used for spoof detection with only 2 linear layer and a relu activation 
class DownStreamTwoLayerClassifier(nn.Module):
    def __init__(self, encoder, input_depth=1024):
        super(DownStreamTwoLayerClassifier, self).__init__()
        self.encoder = encoder  # this should be able to encoder the input(batch_size, 64600) into feature vectors(batch_size, input_depth)
        self.input_depth = input_depth
        self.fc1 = nn.Linear(input_depth, 128) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)  

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.squeeze(1)
        return x
    
# define the downstream classifier used for spoof detection with 3 linear layer and 2 relu activation 
class DownStreamThreeLayerClassifier(nn.Module):
    def __init__(self, encoder, input_depth=1024):
        super(DownStreamThreeLayerClassifier, self).__init__()
        self.encoder = encoder  # this should be able to encoder the input(batch_size, 64600) into feature vectors(batch_size, input_depth)
        self.input_depth = input_depth
        self.fc1 = nn.Linear(input_depth, 128) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)  
        self.fc3 = nn.Linear(64, 2)  

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = x.squeeze(1)
        return x

# RawNet2 Baseline implementation from ASVspoof 2021 baseline, original author: Hemlata Tak
import numpy as np
from collections import OrderedDict


class SincConvBaseline(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    
    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, device,out_channels, kernel_size,in_channels=1,sample_rate=16000,
                 stride=1, padding=0, dilation=1, bias=False, groups=1):

        super(SincConvBaseline,self).__init__()

        if in_channels != 1:
            
            msg = "SincConvBaseline only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate=sample_rate
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
        self.device=device   
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        if bias:
            raise ValueError('SincConvBaseline does not support bias.')
        if groups > 1:
            raise ValueError('SincConvBaseline does not support groups.')
        
        # initialize filterbanks using Mel scale
        NFFT = 512
        f=int(self.sample_rate/2)*np.linspace(0,1,int(NFFT/2)+1)
        fmel=self.to_mel(f)   # Hz to mel conversion
        fmelmax=np.max(fmel)
        fmelmin=np.min(fmel)
        filbandwidthsmel=np.linspace(fmelmin,fmelmax,self.out_channels+1)
        filbandwidthsf=self.to_hz(filbandwidthsmel)  # Mel to Hz conversion
        self.mel=filbandwidthsf
        self.hsupp=torch.arange(-(self.kernel_size-1)/2, (self.kernel_size-1)/2+1)
        self.band_pass=torch.zeros(self.out_channels,self.kernel_size)
    
    def forward(self,x):
        for i in range(len(self.mel)-1):
            fmin=self.mel[i]
            fmax=self.mel[i+1]
            hHigh=(2*fmax/self.sample_rate)*np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow=(2*fmin/self.sample_rate)*np.sinc(2*fmin*self.hsupp/self.sample_rate)
            hideal=hHigh-hLow
            
            self.band_pass[i,:]=Tensor(np.hamming(self.kernel_size))*Tensor(hideal)
        
        band_pass_filter=self.band_pass.to(self.device)

        self.filters = (band_pass_filter).view(self.out_channels, 1, self.kernel_size)
        
        return F.conv1d(x, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)
        
class Residual_blockBaseline(nn.Module):
    def __init__(self, nb_filts, first = False):
        super(Residual_blockBaseline, self).__init__()
        self.first = first
        
        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features = nb_filts[0])
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        
        self.conv1 = nn.Conv1d(in_channels = nb_filts[0],
			out_channels = nb_filts[1],
			kernel_size = 3,
			padding = 1,
			stride = 1)
        
        self.bn2 = nn.BatchNorm1d(num_features = nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels = nb_filts[1],
			out_channels = nb_filts[1],
			padding = 1,
			kernel_size = 3,
			stride = 1)
        
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels = nb_filts[0],
				out_channels = nb_filts[1],
				padding = 0,
				kernel_size = 1,
				stride = 1)
            
        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)
        
    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x
            
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        
        if self.downsample:
            identity = self.conv_downsample(identity)
            
        out += identity
        out = self.mp(out)
        return out

class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)


    def __init__(self, device,out_channels, kernel_size,in_channels=1,sample_rate=16000,
                 stride=1, padding=0, dilation=1, bias=False, groups=1,freq_scale='Mel'):

        super(SincConv,self).__init__()


        if in_channels != 1:
            
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)
        
        self.out_channels = out_channels+1
        self.kernel_size = kernel_size
        self.sample_rate=sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1

        self.device=device   
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')
        
        
        # initialize filterbanks using Mel scale
        NFFT = 512
        f=int(self.sample_rate/2)*np.linspace(0,1,int(NFFT/2)+1)


        if freq_scale == 'Mel':
            fmel=self.to_mel(f) # Hz to mel conversion
            fmelmax=np.max(fmel)
            fmelmin=np.min(fmel)
            filbandwidthsmel=np.linspace(fmelmin,fmelmax,self.out_channels+2)
            filbandwidthsf=self.to_hz(filbandwidthsmel) # Mel to Hz conversion
            self.freq=filbandwidthsf[:self.out_channels]

        elif freq_scale == 'Inverse-mel':
            fmel=self.to_mel(f) # Hz to mel conversion
            fmelmax=np.max(fmel)
            fmelmin=np.min(fmel)
            filbandwidthsmel=np.linspace(fmelmin,fmelmax,self.out_channels+2)
            filbandwidthsf=self.to_hz(filbandwidthsmel) # Mel to Hz conversion
            self.mel=filbandwidthsf[:self.out_channels]
            self.freq=np.abs(np.flip(self.mel)-1) ## invert mel scale

        
        else:
            fmelmax=np.max(f)
            fmelmin=np.min(f)
            filbandwidthsmel=np.linspace(fmelmin,fmelmax,self.out_channels+2)
            self.freq=filbandwidthsmel[:self.out_channels]
        
        self.hsupp=torch.arange(-(self.kernel_size-1)/2, (self.kernel_size-1)/2+1)
        self.band_pass=torch.zeros(self.out_channels-1,self.kernel_size)
    
       
        
    def forward(self,x):
        for i in range(len(self.freq)-1):
            fmin=self.freq[i]
            fmax=self.freq[i+1]
            hHigh=(2*fmax/self.sample_rate)*np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow=(2*fmin/self.sample_rate)*np.sinc(2*fmin*self.hsupp/self.sample_rate)
            hideal=hHigh-hLow
            
            self.band_pass[i,:]=Tensor(np.hamming(self.kernel_size))*Tensor(hideal)
        
        band_pass_filter=self.band_pass.to(self.device)

        self.filters = (band_pass_filter).view(self.out_channels-1, 1, self.kernel_size)
        
        return F.conv1d(x, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)


        
class Residual_block(nn.Module):
    def __init__(self, nb_filts, first = False):
        super(Residual_block, self).__init__()
        self.first = first
        
        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features = nb_filts[0])
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        
        self.conv1 = nn.Conv1d(in_channels = nb_filts[0],
			out_channels = nb_filts[1],
			kernel_size = 3,
			padding = 1,
			stride = 1)
        
        self.bn2 = nn.BatchNorm1d(num_features = nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels = nb_filts[1],
			out_channels = nb_filts[1],
			padding = 1,
			kernel_size = 3,
			stride = 1)
        
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels = nb_filts[0],
				out_channels = nb_filts[1],
				padding = 0,
				kernel_size = 1,
				stride = 1)
            
        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)
        
    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x
            
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        
        if self.downsample:
            identity = self.conv_downsample(identity)
            
        out += identity
        out = self.mp(out)
        return out
    
# SAMO
class SAMOArgs:
    def __init__(self):
        self.seed = 10
        self.path_to_database = "Not initiated"
        self.path_to_protocol = "./samo/protocols"
        self.out_fold = './models/try/'
        self.overwrite = False
        self.enc_dim = 160
        self.num_epochs = 100
        self.batch_size = 32
        self.lr = 0.0001
        self.lr_min = 0.000005
        self.lr_decay = 0.95
        self.interval = 1
        self.scheduler = "cosine2"
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.eps = 1e-8
        self.gpu = "0"
        self.num_workers = 0
        self.loss = "samo"
        self.num_centers = 20
        self.initialize_centers = "one_hot"
        self.m_real = 0.7
        self.m_fake = 0
        self.alpha = 20
        self.continue_training = False
        self.checkpoint = None
        self.test_on_eval = False
        self.final_test = False
        self.test_interval = 5
        self.save_interval = 5
        self.test_only = False
        self.test_model = "./models/anti-spoofing_feat_model.pt"
        self.scoring = None
        self.save_score = None
        self.save_center = False
        self.dp = False
        self.one_hot = False
        self.train_sp = 1
        self.val_sp = 1
        self.target = 1
        self.update_interval = 3
        self.init_center = 1
        self.update_stop = None
        self.center_sampler = "sequential"
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")