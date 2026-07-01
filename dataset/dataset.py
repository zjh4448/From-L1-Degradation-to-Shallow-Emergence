import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import random
import re

from dataset.utils import read_exr_rgb, to_log_domain
from dataset.transforms import PairTransform, ValidationTransform


class N2NDataset(Dataset):
    """Noise2Noise 数据集：sample_0(4spp) → sample_1(32spp)"""

    def __init__(self, data_dir, config, mode='train'):
        self.data_dir = Path(data_dir)
        self.config = config
        self.mode = mode
        self.pairs = []

        if not self.data_dir.exists():
            raise FileNotFoundError(f"目录不存在: {self.data_dir}")

        self._scan_pairs()

        if len(self.pairs) == 0:
            raise ValueError(f"未找到配对数据: {self.data_dir}")

        if mode == 'train':
            self.transform = PairTransform(config)
        else:
            self.transform = ValidationTransform(config)

        print(f"\n{mode.upper()} N2N数据集: {len(self.pairs)} 对")

    def _scan_pairs(self):
        for scene_dir in sorted(self.data_dir.iterdir()):
            if not scene_dir.is_dir():
                continue

            view_dict = {}
            for f in scene_dir.glob("*.exr"):
                m = re.match(r'(view_\d+)_sample_(\d+)', f.stem)
                if m:
                    view_name = m.group(1)
                    sample_id = int(m.group(2))
                    if view_name not in view_dict:
                        view_dict[view_name] = {}
                    view_dict[view_name][sample_id] = f

            for view_name, samples in view_dict.items():
                if 0 in samples and 1 in samples:
                    self.pairs.append({
                        'input': samples[0],
                        'target': samples[1],
                        'scene': scene_dir.name,
                        'view': view_name
                    })

        print(f"  扫描到 {len(self.pairs)} 对 N2N 配对")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx % len(self.pairs)]
        try:
            a = torch.from_numpy(read_exr_rgb(pair['input'])).float()
            b = torch.from_numpy(read_exr_rgb(pair['target'])).float()

            eps = self.config['data']['exr_epsilon']
            a = to_log_domain(a, eps)
            b = to_log_domain(b, eps)

            a, b = self.transform(a, b)

            a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            b = torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)

            return {
                'input': a,
                'target': b,
                'scene': pair['scene'],
                'view': pair['view']
            }
        except Exception:
            return self.__getitem__((idx + 1) % len(self.pairs))


def create_dataloaders(config):
    train_ds = N2NDataset(config['data']['train_dir'], config, mode='train')
    val_ds = N2NDataset(config['data']['val_dir'], config, mode='val')

    bs = config['training']['batch_size']

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True
    )

    print(f"DataLoader: batch={bs}, train_batches={len(train_loader)}, val_batches={len(val_loader)}")
    return train_loader, val_loader