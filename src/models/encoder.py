"""Backbone encoders for continual learning experiments."""

import math

import torch
import torch.nn as nn
import torchvision.models as tv_models


class MLPEncoder(nn.Module):
    """Multi-layer perceptron encoder for MNIST-scale experiments.

    Architecture: input → Linear(hidden) → ReLU → Linear(hidden) → ReLU → Linear(latent_dim)
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


class ResNetEncoder(nn.Module):
    """ResNet-18 encoder for CIFAR-scale experiments.

    Replaces the final FC layer with a linear projection to `latent_dim`.
    Modifies the first conv layer for 32×32 inputs (CIFAR) instead of 224×224.
    """

    def __init__(self, latent_dim: int = 256, pretrained: bool = False, small_input: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        resnet = tv_models.resnet18(
            weights=tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        if small_input:
            # Adapt for 32×32 inputs (CIFAR): smaller kernel, no maxpool
            resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            resnet.maxpool = nn.Identity()

        # Remove original FC, replace with projection to latent_dim
        feat_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()

        self.backbone = resnet
        self.projection = nn.Linear(feat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.projection(features)


class ViTEncoder(nn.Module):
    """Vision Transformer encoder for CIFAR-scale experiments (trained from scratch).

    Uses patch embedding, learnable positional encoding, and a CLS token.
    The CLS token output is projected to latent_dim.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2

        # Patch embedding: conv2d with kernel=stride=patch_size
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            _TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(embed_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Patch embed: (B, C, H, W) → (B, num_patches, embed_dim)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        # Transformer
        x = self.blocks(x)
        x = self.norm(x)
        # CLS token → projection
        return self.projection(x[:, 0])


class _TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.norm1(x)
        x = x + self.attn(x2, x2, x2, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x
