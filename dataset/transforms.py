# dataset/transforms.py
import torch
import random


class PairTransform:
    """对图像对应用相同的变换"""

    def __init__(self, config):
        self.config = config
        self.patch_size = config['training']['patch_size']
        aug = config.get('data_augmentation', {})
        self.hflip_prob = aug.get('hflip_prob', 0.5)
        self.vflip_prob = aug.get('vflip_prob', 0.5)
        self.rotate_90_prob = aug.get('rotate_90_prob', 0.3)
        self.color_jitter = aug.get('color_jitter', False)

    def __call__(self, img_a, img_b):
        _, h, w = img_a.shape

        # 随机裁剪
        if h > self.patch_size and w > self.patch_size:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            img_a = img_a[:, top:top + self.patch_size, left:left + self.patch_size]
            img_b = img_b[:, top:top + self.patch_size, left:left + self.patch_size]

        # 随机水平翻转
        if random.random() < self.hflip_prob:
            img_a = torch.flip(img_a, dims=[2])
            img_b = torch.flip(img_b, dims=[2])

        # 随机垂直翻转
        if random.random() < self.vflip_prob:
            img_a = torch.flip(img_a, dims=[1])
            img_b = torch.flip(img_b, dims=[1])

        # 随机90度旋转
        if random.random() < self.rotate_90_prob:
            k = random.randint(1, 3)
            img_a = torch.rot90(img_a, k, dims=[1, 2])
            img_b = torch.rot90(img_b, k, dims=[1, 2])

        # 颜色扰动（同时应用于两张图，保持配对）
        if self.color_jitter:
            brightness = random.uniform(
                self.config['data_augmentation'].get('brightness_range', [0.8, 1.2])[0],
                self.config['data_augmentation'].get('brightness_range', [0.8, 1.2])[1]
            )
            img_a = img_a * brightness
            img_b = img_b * brightness

        return img_a, img_b


class ValidationTransform:
    """验证集：中心裁剪到 32 的倍数"""

    def __init__(self, config):
        self.patch_size = config['training']['patch_size']

    def __call__(self, img_a, img_b):
        _, h, w = img_a.shape

        # 对齐到 32 的倍数（中心裁剪）
        new_h = h - (h % 32)
        new_w = w - (w % 32)

        if new_h < h or new_w < w:
            top = (h - new_h) // 2
            left = (w - new_w) // 2
            img_a = img_a[:, top:top + new_h, left:left + new_w]
            img_b = img_b[:, top:top + new_h, left:left + new_w]

        return img_a, img_b