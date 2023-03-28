import torch
from torch import nn
from torch.nn import functional as F


class AffineTransform(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, 1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x):
        return self.alpha * x + self.beta


class CommunicationLayer(nn.Module):
    def __init__(self, num_features, num_patches):
        super().__init__()
        self.aff1 = AffineTransform(num_features)
        self.fc1 = nn.Linear(num_patches, num_patches)
        self.aff2 = AffineTransform(num_features)

    def forward(self, x):
        x = self.aff1(x)
        residual = x
        x = self.fc1(x.transpose(1, 2)).transpose(1, 2)
        x = self.aff2(x)
        out = x + residual
        return out


class FeedForward(nn.Module):
    def __init__(self, num_features, expansion_factor):
        super().__init__()
        num_hidden = num_features * expansion_factor
        self.aff1 = AffineTransform(num_features)
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.aff2 = AffineTransform(num_features)

    def forward(self, x):
        x = self.aff1(x)
        residual = x
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.aff2(x)
        out = x + residual
        return out


class ResMLPLayer(nn.Module):
    def __init__(self, num_features, num_patches, expansion_factor):
        super().__init__()
        self.cl = CommunicationLayer(num_features, num_patches)
        self.ff = FeedForward(num_features, expansion_factor)

    def forward(self, x):
        x = self.cl(x)
        out = self.ff(x)
        return out


def check_sizes(image_size, patch_size):
    sqrt_num_patches, remainder = divmod(image_size, patch_size)
    assert remainder == 0, "`image_size` must be divisibe by `patch_size`"
    num_patches = sqrt_num_patches ** 2
    return num_patches


class ResMLP(nn.Module):
    def __init__(
        self,
        image_size=512,
        patch_size=16,
        in_channels=1,
        num_features=384,
        expansion_factor=2,
        num_layers=12,
        num_classes=1000,
        disease_classes=1000
        ):
        num_patches = check_sizes(image_size, patch_size)
        super().__init__()
        self.patcher = nn.Conv2d(
            in_channels, num_features, kernel_size=patch_size, stride=patch_size
        )
        self.mlps = nn.Sequential(
            *[
                ResMLPLayer(num_features, num_patches, expansion_factor)
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(num_features, num_classes)
        self.change_linear = nn.Linear(2000, 2)
        # self.disease_linear = nn.Linear(1000, disease_classes)

    def forward(self, x, x2):
        patches = self.patcher(x)
        patches2 = self.patcher(x2)
        
        batch_size, num_features, _, _ = patches.shape
        
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_features)
        embedding = self.mlps(patches)
        embedding = torch.mean(embedding, dim=1)
        x = self.classifier(embedding)
        
        patches2 = patches2.permute(0, 2, 3, 1)
        patches2 = patches2.view(batch_size, -1, num_features)
        embedding2 = self.mlps(patches2)
        embedding2 = torch.mean(embedding2, dim=1)
        x2 = self.classifier(embedding2)
        
        cat = torch.cat([x,x2],1)
        out = self.change_linear(cat)

        # x1 = self.disease_linear(x)
        # x2 = self.disease_linear(x2)

        return out