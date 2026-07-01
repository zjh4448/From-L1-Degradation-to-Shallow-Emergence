# models/scene_bias.py
import torch
import torch.nn as nn


class SceneBias(nn.Module):
    """从TinyDenoiser浅层特征学习场景空间偏置"""
    def __init__(self, in_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1)
        )

    def forward(self, feat):
        return self.net(feat)