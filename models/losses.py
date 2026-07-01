# models/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientLoss(nn.Module):
    """梯度损失，保持边缘"""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        C = pred.shape[1]
        sx = self.sobel_x.repeat(C, 1, 1, 1)
        sy = self.sobel_y.repeat(C, 1, 1, 1)
        gp_x = F.conv2d(pred, sx, padding=1, groups=C)
        gp_y = F.conv2d(pred, sy, padding=1, groups=C)
        gt_x = F.conv2d(target, sx, padding=1, groups=C)
        gt_y = F.conv2d(target, sy, padding=1, groups=C)
        gp = torch.abs(gp_x) + torch.abs(gp_y)
        gt = torch.abs(gt_x) + torch.abs(gt_y)
        return self.l1(gp, gt)


class MultiscaleL1Loss(nn.Module):
    """多尺度 L1"""

    def __init__(self, scales=(1, 2, 4), weights=(1.0, 0.5, 0.25)):
        super().__init__()
        self.scales = scales
        self.weights = weights
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        loss = 0.0
        for s, w in zip(self.scales, self.weights):
            if s == 1:
                loss += w * self.l1(pred, target)
            else:
                loss += w * self.l1(F.avg_pool2d(pred, s), F.avg_pool2d(target, s))
        return loss


class TotalLoss(nn.Module):
    """可配置的总损失，所有组件独立开关"""

    def __init__(self, config):
        super().__init__()
        loss_cfg = config['loss']

        self.use_channel_weight = loss_cfg.get('use_channel_weight', False)
        if self.use_channel_weight:
            weights = loss_cfg.get('channel_weights', [1.5, 1.0, 2.5])
            self.register_buffer('ch_weight', torch.tensor(weights).view(1, 3, 1, 1))
        else:
            self.register_buffer('ch_weight', torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1))

        self.use_positive_bias = loss_cfg.get('use_positive_bias', False)
        self.positive_bias_weight = loss_cfg.get('positive_bias_weight', 0.3)

        self.use_multiscale = loss_cfg.get('use_multiscale', False)
        if self.use_multiscale:
            scales = loss_cfg.get('multiscale_scales', [1, 2, 4])
            scale_weights = loss_cfg.get('multiscale_scale_weights', [1.0, 0.5, 0.25])
            self.multiscale = MultiscaleL1Loss(scales, scale_weights)
            self.ms_weight = loss_cfg.get('multiscale_weight', 0.1)
            self.ms_start = loss_cfg.get('multiscale_start_epoch', 5)
        else:
            self.multiscale = None

        self.use_gradient = loss_cfg.get('use_gradient', False)
        if self.use_gradient:
            self.gradient = GradientLoss()
            self.grad_weight = loss_cfg.get('gradient_weight', 0.05)
            self.grad_start = loss_cfg.get('gradient_start_epoch', 1)
        else:
            self.gradient = None

    def forward(self, pred, target, epoch, validation=False, prior_info=None):
        loss = (torch.abs(pred - target) * self.ch_weight).mean()

        if validation:
            return loss

        if self.use_positive_bias:
            positive_bias = torch.clamp(pred - target, min=0)
            loss += self.positive_bias_weight * positive_bias.mean()

        if self.use_multiscale and epoch >= self.ms_start:
            loss += self.ms_weight * self.multiscale(pred, target)

        if self.use_gradient and epoch >= self.grad_start:
            loss += self.grad_weight * self.gradient(pred, target)

        return loss