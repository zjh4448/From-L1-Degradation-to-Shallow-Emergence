# duibi.py - 生成对比图 + 全面降噪效果分析（适配TinyDenoiser）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import yaml
import torch

from dataset.utils import read_exr_rgb, write_exr_rgb, visualize_tensor
from dataset.utils import to_log_domain, from_log_domain
from models.unet import TinyDenoiser


def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path, config, device):
    model_config = config['model']
    model = TinyDenoiser(
        in_ch=3,
        out_ch=3,
        features=model_config.get('features', [32, 48, 64])
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 'unknown')
        val_loss = checkpoint.get('val_loss', checkpoint.get('best_val_loss', 'unknown'))
        print(f"Loaded checkpoint - Epoch: {epoch}, Val Loss: {val_loss}")
    elif 'ema_shadow' in checkpoint:
        state_dict = {}
        for key, value in checkpoint['ema_shadow'].items():
            state_dict[key] = value
        print("Loaded EMA shadow weights")
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


# ============================================================
# 分析函数
# ============================================================

def analyze_denoising(noisy, denoised, gt):
    analysis = {}

    min_h = min(noisy.shape[1], denoised.shape[1], gt.shape[1])
    min_w = min(noisy.shape[2], denoised.shape[2], gt.shape[2])
    noisy = noisy[:, :min_h, :min_w]
    denoised = denoised[:, :min_h, :min_w]
    gt = gt[:, :min_h, :min_w]

    data_range = gt.max() - gt.min()
    if data_range <= 0:
        data_range = 1.0

    mse_noisy = np.mean((noisy - gt) ** 2)
    mse_denoised = np.mean((denoised - gt) ** 2)
    mae_noisy = np.mean(np.abs(noisy - gt))
    mae_denoised = np.mean(np.abs(denoised - gt))

    psnr_noisy = 20 * np.log10(data_range) - 10 * np.log10(mse_noisy + 1e-8)
    psnr_denoised = 20 * np.log10(data_range) - 10 * np.log10(mse_denoised + 1e-8)

    analysis['psnr_noisy'] = psnr_noisy
    analysis['psnr_denoised'] = psnr_denoised
    analysis['psnr_improvement'] = psnr_denoised - psnr_noisy
    analysis['mse_noisy'] = mse_noisy
    analysis['mse_denoised'] = mse_denoised
    analysis['mae_noisy'] = mae_noisy
    analysis['mae_denoised'] = mae_denoised

    channel_names = ['R', 'G', 'B']
    channel_psnr = {}
    channel_mse = {}
    for i, ch in enumerate(channel_names):
        mse_n = np.mean((noisy[i] - gt[i]) ** 2)
        mse_d = np.mean((denoised[i] - gt[i]) ** 2)
        channel_mse[ch] = {'noisy': mse_n, 'denoised': mse_d, 'reduction': (mse_n - mse_d) / (mse_n + 1e-8) * 100}
        channel_psnr[ch] = {
            'noisy': 20 * np.log10(data_range) - 10 * np.log10(mse_n + 1e-8),
            'denoised': 20 * np.log10(data_range) - 10 * np.log10(mse_d + 1e-8)
        }
    analysis['channel_mse'] = channel_mse
    analysis['channel_psnr'] = channel_psnr

    error_map_noisy = np.mean((noisy - gt) ** 2, axis=0)
    error_map_denoised = np.mean((denoised - gt) ** 2, axis=0)

    luminance = np.mean(gt, axis=0)
    bright_mask = luminance > np.percentile(luminance, 80)
    dark_mask = luminance < np.percentile(luminance, 20)
    mid_mask = ~bright_mask & ~dark_mask

    analysis['bright_region'] = {
        'mse_noisy': float(np.mean(error_map_noisy[bright_mask])),
        'mse_denoised': float(np.mean(error_map_denoised[bright_mask])),
        'pixel_pct': float(bright_mask.mean() * 100)
    }
    analysis['mid_region'] = {
        'mse_noisy': float(np.mean(error_map_noisy[mid_mask])),
        'mse_denoised': float(np.mean(error_map_denoised[mid_mask])),
        'pixel_pct': float(mid_mask.mean() * 100)
    }
    analysis['dark_region'] = {
        'mse_noisy': float(np.mean(error_map_noisy[dark_mask])),
        'mse_denoised': float(np.mean(error_map_denoised[dark_mask])),
        'pixel_pct': float(dark_mask.mean() * 100)
    }

    from scipy import ndimage
    grad_gt = np.abs(ndimage.sobel(gt[0])) + np.abs(ndimage.sobel(gt[1])) + np.abs(ndimage.sobel(gt[2]))
    edge_mask = grad_gt > np.percentile(grad_gt, 90)
    smooth_mask = grad_gt < np.percentile(grad_gt, 10)

    analysis['edge_region'] = {
        'mse_noisy': float(np.mean(error_map_noisy[edge_mask])),
        'mse_denoised': float(np.mean(error_map_denoised[edge_mask])),
        'pixel_pct': float(edge_mask.mean() * 100)
    }
    analysis['smooth_region'] = {
        'mse_noisy': float(np.mean(error_map_noisy[smooth_mask])),
        'mse_denoised': float(np.mean(error_map_denoised[smooth_mask])),
        'pixel_pct': float(smooth_mask.mean() * 100)
    }

    if analysis['edge_region']['mse_noisy'] > 0:
        analysis['edge_improvement'] = (
            (analysis['edge_region']['mse_noisy'] - analysis['edge_region']['mse_denoised'])
            / analysis['edge_region']['mse_noisy'] * 100
        )
    else:
        analysis['edge_improvement'] = 0

    analysis['pixel_stats'] = {
        'noisy': {'min': float(noisy.min()), 'max': float(noisy.max()),
                   'mean': float(noisy.mean()), 'std': float(noisy.std())},
        'denoised': {'min': float(denoised.min()), 'max': float(denoised.max()),
                      'mean': float(denoised.mean()), 'std': float(denoised.std())},
        'gt': {'min': float(gt.min()), 'max': float(gt.max()),
               'mean': float(gt.mean()), 'std': float(gt.std())},
    }

    var_ratio = denoised.std() / (gt.std() + 1e-8)
    analysis['variance_ratio'] = float(var_ratio)
    analysis['over_smoothing'] = var_ratio < 0.7

    highlight_threshold = np.percentile(gt, 95)
    highlight_mask = gt > highlight_threshold
    if highlight_mask.sum() > 0:
        analysis['highlight'] = {
            'gt_max': float(gt[highlight_mask].max()),
            'denoised_max': float(denoised[highlight_mask].max()),
            'ratio': float(denoised[highlight_mask].max() / (gt[highlight_mask].max() + 1e-8)),
            'mae': float(np.mean(np.abs(denoised[highlight_mask] - gt[highlight_mask])))
        }
    else:
        analysis['highlight'] = None

    suggestions = []
    if analysis['psnr_improvement'] < 2:
        suggestions.append("PSNR提升不足2dB，模型可能未充分训练或数据不匹配")
    elif analysis['psnr_improvement'] < 5:
        suggestions.append("PSNR提升2-5dB，有改善但仍有优化空间")
    else:
        suggestions.append("PSNR提升>5dB，降噪效果显著")

    if analysis.get('over_smoothing'):
        suggestions.append(f"检测到过平滑：降噪图方差仅为GT的{var_ratio*100:.0f}%")

    if analysis.get('highlight') and analysis['highlight']['ratio'] < 0.5:
        suggestions.append(f"高光恢复差：降噪后高光仅为GT的{analysis['highlight']['ratio']*100:.0f}%")

    edge_imp = analysis.get('edge_improvement', 0)
    if edge_imp < 0:
        suggestions.append(f"边缘区域误差增大{-edge_imp:.1f}%，边缘被过度模糊")
    elif edge_imp < 20:
        suggestions.append(f"边缘改善有限({edge_imp:.1f}%)，可增大梯度损失权重")

    worst_ch = max(channel_mse, key=lambda k: channel_mse[k]['reduction'])
    if channel_mse[worst_ch]['reduction'] < 10:
        suggestions.append(f"{worst_ch}通道改善最小({channel_mse[worst_ch]['reduction']:.1f}%)")

    analysis['suggestions'] = suggestions
    return analysis, error_map_noisy, error_map_denoised, luminance


def print_analysis(analysis):
    print("\n" + "=" * 60)
    print("降噪效果全面分析报告")
    print("=" * 60)

    print(f"\n-- 基础指标 --")
    print(f"  PSNR:  噪声 {analysis['psnr_noisy']:.2f} dB -> 降噪 {analysis['psnr_denoised']:.2f} dB "
          f"(提升 +{analysis['psnr_improvement']:.2f} dB)")
    print(f"  MSE:   噪声 {analysis['mse_noisy']:.4f} -> 降噪 {analysis['mse_denoised']:.4f}")
    print(f"  MAE:   噪声 {analysis['mae_noisy']:.4f} -> 降噪 {analysis['mae_denoised']:.4f}")

    print(f"\n-- 逐通道分析 --")
    for ch in ['R', 'G', 'B']:
        c = analysis['channel_psnr'][ch]
        m = analysis['channel_mse'][ch]
        print(f"  {ch}: PSNR {c['noisy']:.2f} -> {c['denoised']:.2f} dB | MSE减少 {m['reduction']:.1f}%")

    print(f"\n-- 亮度区域分析 --")
    for region in ['bright_region', 'mid_region', 'dark_region']:
        r = analysis[region]
        imp = (r['mse_noisy'] - r['mse_denoised']) / (r['mse_noisy'] + 1e-8) * 100
        label = region.replace('_region', '').capitalize()
        print(f"  {label} ({r['pixel_pct']:.1f}%): MSE {r['mse_noisy']:.4f} -> {r['mse_denoised']:.4f} ({imp:+.1f}%)")

    print(f"\n-- 边缘保持分析 --")
    e = analysis['edge_region']
    s = analysis['smooth_region']
    print(f"  边缘 ({e['pixel_pct']:.1f}%): MSE {e['mse_noisy']:.4f} -> {e['mse_denoised']:.4f}")
    print(f"  平滑 ({s['pixel_pct']:.1f}%): MSE {s['mse_noisy']:.4f} -> {s['mse_denoised']:.4f}")
    print(f"  边缘改善: {analysis['edge_improvement']:.1f}%")

    print(f"\n-- 过平滑检测 --")
    print(f"  方差比 (降噪/GT): {analysis['variance_ratio']:.3f}")
    if analysis['over_smoothing']:
        print(f"  检测到过平滑！")

    if analysis['highlight']:
        h = analysis['highlight']
        print(f"\n-- 高光恢复 --")
        print(f"  GT高光最大值: {h['gt_max']:.3f}")
        print(f"  降噪高光最大值: {h['denoised_max']:.3f}")
        print(f"  恢复率: {h['ratio']*100:.1f}%")

    if analysis['suggestions']:
        print(f"\n-- 改进建议 --")
        for s in analysis['suggestions']:
            print(f"  {s}")

    print("=" * 60)


def generate_comparison_with_gt(noisy_path, denoised_path, gt_path, output_path):
    print(f"\n读取图像...")
    noisy = read_exr_rgb(str(noisy_path))
    denoised = read_exr_rgb(str(denoised_path))
    gt = read_exr_rgb(str(gt_path))

    print(f"  噪声图: shape={noisy.shape}, range=[{noisy.min():.3f}, {noisy.max():.3f}]")
    print(f"  降噪图: shape={denoised.shape}, range=[{denoised.min():.3f}, {denoised.max():.3f}]")
    print(f"  GT:     shape={gt.shape}, range=[{gt.min():.3f}, {gt.max():.3f}]")

    min_h = min(noisy.shape[1], denoised.shape[1], gt.shape[1])
    min_w = min(noisy.shape[2], denoised.shape[2], gt.shape[2])
    noisy = noisy[:, :min_h, :min_w]
    denoised = denoised[:, :min_h, :min_w]
    gt = gt[:, :min_h, :min_w]

    analysis, error_map_noisy, error_map_denoised, luminance = analyze_denoising(noisy, denoised, gt)
    print_analysis(analysis)

    noisy_vis = visualize_tensor(noisy)
    denoised_vis = visualize_tensor(denoised)
    gt_vis = visualize_tensor(gt)

    error_max = max(error_map_noisy.max(), error_map_denoised.max(), 0.01)

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))

    axes[0, 0].imshow(noisy_vis)
    axes[0, 0].set_title('Noisy Input', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(denoised_vis)
    axes[0, 1].set_title('Denoised Output', fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(gt_vis)
    axes[0, 2].set_title('Ground Truth', fontsize=13, fontweight='bold')
    axes[0, 2].axis('off')

    residual = denoised_vis - gt_vis
    res_max = max(abs(residual).max(), 0.05)
    im_res = axes[0, 3].imshow(residual, cmap='RdBu_r', vmin=-res_max, vmax=res_max)
    axes[0, 3].set_title('Residual (Denoised - GT)', fontsize=13, fontweight='bold')
    axes[0, 3].axis('off')
    plt.colorbar(im_res, ax=axes[0, 3], fraction=0.046, pad=0.04)

    im1 = axes[1, 0].imshow(error_map_noisy, cmap='hot', vmin=0, vmax=error_max)
    axes[1, 0].set_title(f'MSE Error: Noisy ({error_map_noisy.mean():.4f})', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im2 = axes[1, 1].imshow(error_map_denoised, cmap='hot', vmin=0, vmax=error_max)
    axes[1, 1].set_title(f'MSE Error: Denoised ({error_map_denoised.mean():.4f})', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    error_diff = error_map_noisy - error_map_denoised
    diff_max = max(abs(error_diff).max(), 0.01)
    im3 = axes[1, 2].imshow(error_diff, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
    axes[1, 2].set_title('Error Improvement\n(Blue=Better, Red=Worse)', fontsize=12)
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)

    im4 = axes[1, 3].imshow(luminance, cmap='viridis')
    axes[1, 3].set_title('Luminance (GT)', fontsize=12)
    axes[1, 3].axis('off')
    plt.colorbar(im4, ax=axes[1, 3], fraction=0.046, pad=0.04)

    ax_metrics = axes[2, 0]
    ax_metrics.axis('off')

    s = analysis['suggestions']
    sug_text = '\n'.join(f"  {x}" for x in s) if s else '  None'

    metrics_text = (
        f"METRICS\n"
        f"{'-'*30}\n"
        f"PSNR:  {analysis['psnr_noisy']:.2f} -> {analysis['psnr_denoised']:.2f} dB\n"
        f"       (+{analysis['psnr_improvement']:.2f} dB)\n\n"
        f"MSE:   {analysis['mse_noisy']:.4f} -> {analysis['mse_denoised']:.4f}\n"
        f"MAE:   {analysis['mae_noisy']:.4f} -> {analysis['mae_denoised']:.4f}\n\n"
        f"Over-smoothing: {'YES' if analysis['over_smoothing'] else 'NO'}\n"
        f"Variance Ratio: {analysis['variance_ratio']:.3f}\n\n"
        f"SUGGESTIONS\n"
        f"{'-'*30}\n"
        f"{sug_text}"
    )
    ax_metrics.text(0.02, 0.98, metrics_text, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    transform=ax_metrics.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    channels = ['R', 'G', 'B']
    for i, ch in enumerate(channels):
        ax = axes[2, i+1]
        cm = analysis['channel_mse'][ch]
        bars = ax.bar(['Noisy', 'Denoised'], [cm['noisy'], cm['denoised']],
                       color=['#e74c3c', '#2ecc71'], alpha=0.8)
        ax.set_title(f'{ch} Channel MSE', fontsize=11)
        ax.set_ylabel('MSE')
        for bar, val in zip(bars, [cm['noisy'], cm['denoised']]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                    f'{val:.4f}', ha='center', fontsize=9)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n对比图已保存: {output_path}")
    return analysis


def generate_comparison_single(noisy_path, denoised_path, output_path):
    print(f"\n读取图像...")
    noisy = read_exr_rgb(str(noisy_path))
    denoised = read_exr_rgb(str(denoised_path))

    if noisy.shape != denoised.shape:
        min_h = min(noisy.shape[1], denoised.shape[1])
        min_w = min(noisy.shape[2], denoised.shape[2])
        noisy = noisy[:, :min_h, :min_w]
        denoised = denoised[:, :min_h, :min_w]

    noisy_vis = visualize_tensor(noisy)
    denoised_vis = visualize_tensor(denoised)

    diff = np.abs(noisy_vis - denoised_vis)
    diff_max = max(diff.max(), 0.1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(noisy_vis)
    axes[0].set_title('Noisy Input', fontsize=14)
    axes[0].axis('off')
    axes[1].imshow(denoised_vis)
    axes[1].set_title('Denoised Output', fontsize=14)
    axes[1].axis('off')
    im = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=diff_max)
    axes[2].set_title('|Difference|', fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)

    mse = np.mean((noisy - denoised) ** 2)
    print(f"对比图已保存: {output_path}")
    return {'mse': float(mse)}


def main():
    parser = argparse.ArgumentParser(description='Generate comparison images with analysis')
    parser.add_argument('--noisy', type=str, required=True, help='Noisy EXR file')
    parser.add_argument('--denoised', type=str, default=None, help='Denoised EXR file')
    parser.add_argument('--gt', type=str, default=None, help='Ground truth EXR file')
    parser.add_argument('--output', type=str, default='comparison.png')
    parser.add_argument('--model', type=str, default=None, help='Model checkpoint for live inference')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    print("=" * 55)
    print("生成降噪效果对比图 + 全面分析")
    print("=" * 55)

    noisy_path = Path(args.noisy)
    if not noisy_path.exists():
        print(f"Error: Noisy image not found: {args.noisy}")
        return

    if args.denoised:
        denoised_path = Path(args.denoised)
        if not denoised_path.exists():
            print(f"Error: Denoised image not found: {args.denoised}")
            return
    elif args.model:
        config = load_config(args.config)
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f"Loading model: {args.model}")
        model = load_model(args.model, config, device)

        img = read_exr_rgb(str(noisy_path))
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        img_log = to_log_domain(img_tensor).to(device)

        with torch.no_grad():
            pred_log = model(img_log)
        pred = from_log_domain(pred_log).squeeze(0).cpu().numpy()

        denoised_path = Path('temp_denoised.exr')
        write_exr_rgb(str(denoised_path), pred)
        print(f"Temporary denoised: {denoised_path}")
    else:
        print("Error: Provide --denoised or --model")
        return

    if args.gt:
        gt_path = Path(args.gt)
        if gt_path.exists():
            metrics = generate_comparison_with_gt(
                str(noisy_path), str(denoised_path), str(gt_path),
                args.output
            )
        else:
            print(f"Warning: GT not found, comparing noisy vs denoised only")
            metrics = generate_comparison_single(
                str(noisy_path), str(denoised_path), args.output
            )
    else:
        metrics = generate_comparison_single(
            str(noisy_path), str(denoised_path), args.output
        )

    if args.model and not args.denoised:
        denoised_path.unlink(missing_ok=True)

    print(f"\n查看对比图: {Path(args.output).absolute()}")


if __name__ == "__main__":
    main()