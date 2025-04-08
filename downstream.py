import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Union

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