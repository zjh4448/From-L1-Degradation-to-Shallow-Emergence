# train.py - 分阶段训练版
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import signal
import gc
import sys

from dataset.dataset import create_dataloaders
from models.unet import TinyDenoiser, count_parameters, count_parameters_by_module
from models.losses import TotalLoss
from models.ema import EMA

should_exit = False


def signal_handler(signum, frame):
    global should_exit
    print(f"\n信号 {signum}，保存后退出...")
    should_exit = True


signal.signal(signal.SIGINT, signal_handler)


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_checkpoint(state, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filename)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_metrics(output, target, metrics_list):
    results = {}
    for m in metrics_list:
        if m == 'psnr':
            mse = torch.mean((output - target) ** 2, dim=[1, 2, 3])
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
            results['psnr'] = psnr.mean().item()
        elif m == 'ssim':
            c1, c2 = 0.01 ** 2, 0.03 ** 2
            mu_x = torch.mean(output, dim=[1, 2, 3], keepdim=True)
            mu_y = torch.mean(target, dim=[1, 2, 3], keepdim=True)
            sx = torch.var(output, dim=[1, 2, 3], keepdim=True)
            sy = torch.var(target, dim=[1, 2, 3], keepdim=True)
            sxy = torch.mean((output - mu_x) * (target - mu_y), dim=[1, 2, 3], keepdim=True)
            ssim = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / (
                    (mu_x ** 2 + mu_y ** 2 + c1) * (sx + sy + c2))
            results['ssim'] = ssim.mean().item()
    return results


def set_requires_grad(model, requires_grad, module_prefix='scene_bias'):
    """冻结/解冻指定模块"""
    for name, param in model.named_parameters():
        if module_prefix in name:
            param.requires_grad = requires_grad
        else:
            param.requires_grad = not requires_grad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"配置: {args.config}")

    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_cfg = config['output']
    for d in [output_cfg['checkpoint_dir'], output_cfg['log_dir'], output_cfg['vis_dir']]:
        Path(d).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    print("加载数据...")
    train_loader, val_loader = create_dataloaders(config)

    use_scene_bias = config['model'].get('use_scene_bias', False)
    scene_bias_config = config['model'].get('scene_bias_config', None)

    model = TinyDenoiser(
        features=config['model'].get('features', [32, 48, 64]),
        use_scene_bias=use_scene_bias,
        scene_bias_config=scene_bias_config
    ).to(device)

    param_stats = count_parameters_by_module(model)
    print(f"模型参数:")
    for module_name, params in param_stats.items():
        print(f"  {module_name}: {params:,}")
    print(f"  总计: {count_parameters(model):,}")

    ema = EMA(model, decay=0.999)
    ema.to(device)
    criterion = TotalLoss(config).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=config['training']['lr_factor'],
        patience=config['training']['lr_patience'],
        min_lr=config['training']['min_lr']
    )

    writer = SummaryWriter(output_cfg['log_dir'])
    metrics_list = config.get('validation', {}).get('metrics', [])

    start_epoch = 0
    best_val_loss = float('inf')
    train_losses, val_losses, lrs = [], [], []
    no_improve = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        train_losses = ckpt.get('train_losses', [])
        val_losses = ckpt.get('val_losses', [])
        no_improve = ckpt.get('no_improve', 0)
        ema.to(device)

    total_epochs = config['training']['epochs']
    phase1_epochs = config['training'].get('phase1_epochs', 30)  # 阶段一epoch数
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    accumulation = config['training'].get('gradient_accumulation', 1)
    early_stop_patience = config['validation'].get('early_stop_patience', 25)

    print(f"\n{'=' * 60}")
    print(f"训练: {start_epoch + 1}→{total_epochs} epochs")
    print(f"batch={config['training']['batch_size']}, lr={config['training']['lr']}")
    print(f"SceneBias: {'✓' if use_scene_bias else '✗'}")
    print(f"阶段一(1-{phase1_epochs}): 训练TinyDenoiser, SceneBias冻结")
    print(f"阶段二({phase1_epochs+1}-{total_epochs}): 训练SceneBias, TinyDenoiser冻结")
    print(f"{'=' * 60}\n")

    for epoch in range(start_epoch + 1, total_epochs + 1):
        if should_exit:
            break

        # 分阶段冻结/解冻
        if use_scene_bias:
            if epoch <= phase1_epochs:
                # 阶段一：训练TinyDenoiser，冻结SceneBias
                set_requires_grad(model, requires_grad=False, module_prefix='scene_bias')
            else:
                # 阶段二：冻结TinyDenoiser，训练SceneBias
                set_requires_grad(model, requires_grad=True, module_prefix='scene_bias')

        model.train()
        total_loss = 0
        nb = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        optimizer.zero_grad()

        if warmup_epochs > 0 and epoch <= warmup_epochs:
            lr = config['training']['lr'] * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        for bi, batch in enumerate(pbar):
            x = batch['input'].to(device)
            y = batch['target'].to(device)

            pred = model(x)
            loss = criterion(pred, y, epoch, validation=False) / accumulation
            loss.backward()

            if (bi + 1) % accumulation == 0 or (bi + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation
            nb += 1
            pbar.set_postfix({'loss': f'{loss.item() * accumulation:.4f}'})

        ema.update()
        train_loss = total_loss / max(nb, 1)

        model.eval()
        val_loss = 0
        val_metrics_all = {m: [] for m in metrics_list}
        with torch.no_grad():
            for batch in val_loader:
                x = batch['input'].to(device)
                y = batch['target'].to(device)
                pred = model(x)
                val_loss += criterion(pred, y, epoch, validation=True).item()
                if metrics_list:
                    bm = compute_metrics(pred, y, metrics_list)
                    for k, v in bm.items():
                        val_metrics_all[k].append(v)

        val_loss /= max(len(val_loader), 1)
        val_metrics = {k: np.mean(v) for k, v in val_metrics_all.items() if v}

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lrs.append(current_lr)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'Metrics/{k}', v, epoch)

        phase = "阶段一(U-Net)" if epoch <= phase1_epochs else "阶段二(SceneBias)"
        print(f"Epoch {epoch} [{phase}]: Train={train_loss:.4f}, Val={val_loss:.4f}, "
              f"LR={current_lr:.2e} | "
              + ' | '.join(f"{k.upper()}={v:.4f}" for k, v in val_metrics.items()))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            save_checkpoint({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_shadow': ema.shadow, 'best_val_loss': best_val_loss,
                'train_losses': train_losses, 'val_losses': val_losses,
                'lrs': lrs, 'no_improve': no_improve,
            }, Path(output_cfg['checkpoint_dir']) / 'best_model.pth')

            ema.apply_shadow()
            save_checkpoint({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
            }, Path(output_cfg['checkpoint_dir']) / 'best_model_ema.pth')
            ema.restore()
            print(f"  ★ 新最佳模型! (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1

        save_every = output_cfg.get('save_every', 10)
        if epoch % save_every == 0:
            save_checkpoint({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_shadow': ema.shadow, 'best_val_loss': best_val_loss,
            }, Path(output_cfg['checkpoint_dir']) / f'epoch_{epoch:03d}.pth')

        if no_improve >= early_stop_patience and epoch > warmup_epochs:
            print(f"\n早停: epoch {epoch}, {early_stop_patience} epochs 无改善")
            break

    save_checkpoint({
        'epoch': total_epochs, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ema_shadow': ema.shadow, 'best_val_loss': best_val_loss,
        'train_losses': train_losses, 'val_losses': val_losses, 'lrs': lrs,
    }, Path(output_cfg['checkpoint_dir']) / 'final_model.pth')

    ema.apply_shadow()
    save_checkpoint({
        'epoch': total_epochs, 'model_state_dict': model.state_dict(),
        'best_val_loss': best_val_loss,
    }, Path(output_cfg['checkpoint_dir']) / 'final_model_ema.pth')
    ema.restore()

    writer.close()
    print(f"\n训练完成！最佳 Val Loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()