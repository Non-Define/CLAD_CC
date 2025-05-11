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
        print("[Debug] Input to encoder_q (q):", x_q.shape)
        q = self.encoder_q(x_q) # queries: NxC
        print("[Debug] Output from encoder_q:", q.shape)
        
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
            print("[Debug] Input to encoder_k (k):", x_k.shape)
            k = self.encoder_k(x_k)
            print("[Debug] Output from encoder_k:", k.shape)
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