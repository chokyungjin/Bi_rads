import numpy as np
import torch
from einops.layers.torch import Rearrange
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):

        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x


class MLPMixer(nn.Module):

    def __init__(self, 
                 in_channels=1, 
                 dim=512, 
                 num_classes=1000, 
                 patch_size=16, 
                 image_size=512, 
                 depth=12, 
                 token_dim=256, 
                 channel_dim=2048,
                 num_label=1000,
                 disease_classes=1000):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch =  (image_size// patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )
        
        self.change_linear = nn.Linear(2000,2)
        self.disease_linear = nn.Linear(1000, disease_classes)

    def forward(self, x, x2):


        x = self.to_patch_embedding(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        x = self.mlp_head(x)
        
        x2 = self.to_patch_embedding(x2)
        for mixer_block in self.mixer_blocks:
            x2 = mixer_block(x2)
        x2 = self.layer_norm(x2)
        x2 = x2.mean(dim=1)
        x2 = self.mlp_head(x2)
        
        cat = torch.cat([x,x2],1)
        out = self.change_linear(cat)
        
        return out