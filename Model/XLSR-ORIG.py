# Define Model (XLSR-ORIG)

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
            nn.ReLU(),
            nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=(3, 3), padding=(1, 1)),
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
        