'''
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# FeatureExtractor만 사용 (tokenizer 불필요)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
'''

import random
import torch
import numpy as np
from Model.XLSR_ORIG import ConvLayers


random_array = np.random.randn(201, 1024).astype(np.float32)
input_tensor = torch.tensor(random_array).unsqueeze(0)

model = ConvLayers(input_dim=1024, proj_dim=256, conv_channels=32)
output = model(input_tensor)

print("Output shape:", output.shape)
