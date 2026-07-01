# models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyBlock(nn.Module):
    """双层深度可分离卷积块"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw1 = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.dw2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False)
        self.pw2 = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.bn1(self.pw1(self.dw1(x))))
        out = self.bn2(self.pw2(self.dw2(out)))
        return self.relu(out + residual)


class TinyDenoiser(nn.Module):
    """极轻量U-Net降噪网络 + 可选SceneBias场景偏置（浅层特征驱动）"""

    def __init__(self, in_ch=3, out_ch=3, features=None,
                 use_scene_bias=False, scene_bias_config=None):
        super().__init__()
        if features is None:
            features = [32, 48, 64]
        f0, f1, f2 = features

        self.use_scene_bias = use_scene_bias

        self.enc1 = TinyBlock(in_ch, f0)
        self.enc2 = TinyBlock(f0, f1)
        self.enc3 = TinyBlock(f1, f2)
        self.down = nn.MaxPool2d(2)
        self.bottleneck = TinyBlock(f2, f2)

        self.up3 = nn.ConvTranspose2d(f2, f2, 2, 2)
        self.dec3 = TinyBlock(f2 + f2, f1)
        self.up2 = nn.ConvTranspose2d(f1, f1, 2, 2)
        self.dec2 = TinyBlock(f1 + f1, f0)
        self.up1 = nn.ConvTranspose2d(f0, f0, 2, 2)
        self.dec1 = TinyBlock(f0 + f0, f0)

        self.out_conv = nn.Conv2d(f0, out_ch, 3, padding=1)

        if use_scene_bias:
            from models.scene_bias import SceneBias
            cfg = scene_bias_config or {}
            self.scene_bias = SceneBias(
                in_ch=cfg.get('in_ch', 32)
            )
            self.bias_scale = cfg.get('bias_scale', 1.0)
        else:
            self.scene_bias = None

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        e3 = self.enc3(self.down(e2))
        b = self.bottleneck(self.down(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        base = self.out_conv(d1)
        denoised_init = base + x

        if self.scene_bias is not None:
            bias = self.scene_bias(e1)
            w = torch.clamp(1.0 - bias * self.bias_scale, 0, 1)
            noise_estimate = x - denoised_init
            out = x - w * noise_estimate
            return out

        return denoised_init


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_by_module(model):
    stats = {}
    unet_params = 0
    bias_params = 0

    for name, param in model.named_parameters():
        if 'scene_bias' in name:
            bias_params += param.numel()
        else:
            unet_params += param.numel()

    stats['U-Net'] = unet_params
    if bias_params > 0:
        stats['SceneBias'] = bias_params
    stats['Total'] = unet_params + bias_params
    return stats