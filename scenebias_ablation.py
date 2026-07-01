import torch
import numpy as np
import sys

sys.path.insert(0, '.')
from pathlib import Path
import yaml
from dataset.utils import read_exr_rgb, to_log_domain, from_log_domain
from models.unet import TinyDenoiser
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device('cuda')
base_dir = Path('C:/Users/lenovo/Desktop/ai_random/ai_w')

# ============================================================
# 加载模型
# ============================================================
# 纯TinyDenoiser（D_sponza，独立训练30轮）
with open(base_dir / 'config_D_sponza.yaml', 'r', encoding='utf-8') as f:
    config_d = yaml.safe_load(f)
eps = config_d['data']['exr_epsilon']

ckpt_d = torch.load(base_dir / 'output_n2n/D_sponza/checkpoints/best_model.pth', map_location=device, weights_only=True)
model_d = TinyDenoiser(features=[32, 48, 64], use_scene_bias=False).cuda()
model_d.load_state_dict(ckpt_d['model_state_dict'])
model_d.eval()

# TinyDenoiser + SceneBias（F配置，分阶段训练60轮）
with open(base_dir / 'config_E.yaml', 'r', encoding='utf-8') as f:
    config_f = yaml.safe_load(f)

ckpt_f = torch.load(base_dir / 'output_n2n/F/checkpoints/best_model.pth', map_location=device, weights_only=True)
model_f = TinyDenoiser(features=[32, 48, 64], use_scene_bias=True,
                       scene_bias_config=config_f['model']['scene_bias_config']).cuda()
model_f.load_state_dict(ckpt_f['model_state_dict'])
model_f.eval()

# ============================================================
# 测试多个视角
# ============================================================
test_views = ['center', 'corner', 'left', 'right', 'up']
results_d_log = []
results_f_log = []
results_d_linear = []
results_f_linear = []
bias_stats = []

for view in test_views:
    img_4 = read_exr_rgb(str(base_dir / f'dataset/sponza_test/{view}/spp_004.exr'))
    img_gt = read_exr_rgb(str(base_dir / f'dataset/sponza_test/{view}/ground_truth.exr'))
    x = to_log_domain(torch.from_numpy(img_4).float().unsqueeze(0), eps).cuda()
    gt_log = to_log_domain(torch.from_numpy(img_gt).float().unsqueeze(0), eps).cuda()

    with torch.no_grad():
        pred_d = model_d(x)
        pred_f = model_f(x)
        e1 = model_f.enc1(x)
        bias = model_f.scene_bias(e1)

        # log域PSNR（参考）
        mse_d_log = ((pred_d - gt_log) ** 2).mean().item()
        mse_f_log = ((pred_f - gt_log) ** 2).mean().item()
        results_d_log.append(20 * np.log10(1.0 / np.sqrt(mse_d_log + 1e-8)))
        results_f_log.append(20 * np.log10(1.0 / np.sqrt(mse_f_log + 1e-8)))

        # 线性域PSNR（论文使用）
        pred_d_linear = from_log_domain(pred_d.cpu(), eps).squeeze(0).numpy()
        pred_f_linear = from_log_domain(pred_f.cpu(), eps).squeeze(0).numpy()
        dr = img_gt.max() - img_gt.min()
        mse_d_linear = np.mean((pred_d_linear - img_gt) ** 2)
        mse_f_linear = np.mean((pred_f_linear - img_gt) ** 2)
        results_d_linear.append(20 * np.log10(dr / np.sqrt(mse_d_linear + 1e-8)))
        results_f_linear.append(20 * np.log10(dr / np.sqrt(mse_f_linear + 1e-8)))

        # Bias统计
        bias_np = bias.cpu().squeeze().numpy()
        bias_stats.append({
            'view': view,
            'mean': bias_np.mean(),
            'std': bias_np.std(),
            'min': bias_np.min(),
            'max': bias_np.max()
        })

        # 误差与bias的空间相关性（center视角）
        if view == 'center':
            error_d = (pred_d - gt_log).abs().mean(dim=1, keepdim=True).cpu().squeeze().numpy()
            error_f = (pred_f - gt_log).abs().mean(dim=1, keepdim=True).cpu().squeeze().numpy()
            corr_db = np.corrcoef(error_d.flatten(), bias_np.flatten())[0, 1]
            corr_fb = np.corrcoef(error_f.flatten(), bias_np.flatten())[0, 1]
            bias_center = bias_np
            pred_d_center = pred_d_linear
            pred_f_center = pred_f_linear
            error_d_center = error_d
            error_f_center = error_f

# ============================================================
# 定量结果输出（线性域）
# ============================================================
print("=" * 60)
print("SceneBias Ablation Study — Quantitative Results (Linear Domain)")
print("=" * 60)
print(f"\n{'View':<10} {'Pure TinyDenoiser':<20} {'+ SceneBias':<20} {'Delta':<10}")
print("-" * 60)
for i, view in enumerate(test_views):
    delta = results_f_linear[i] - results_d_linear[i]
    print(f"{view:<10} {results_d_linear[i]:>18.2f} dB {results_f_linear[i]:>18.2f} dB {delta:>+8.2f} dB")

print(f"\n{'Average':<10} {np.mean(results_d_linear):>18.2f} dB {np.mean(results_f_linear):>18.2f} dB {np.mean(results_f_linear) - np.mean(results_d_linear):>+8.2f} dB")
print(f"{'Std':<10} {np.std(results_d_linear):>18.2f} dB {np.std(results_f_linear):>18.2f} dB")

print(f"\n{'=' * 60}")
print("Bias Spatial Statistics (per view)")
print(f"{'=' * 60}")
print(f"{'View':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("-" * 50)
for s in bias_stats:
    print(f"{s['view']:<10} {s['mean']:>10.4f} {s['std']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f}")

print(f"\n{'=' * 60}")
print("Bias-Error Spatial Correlation (center view)")
print(f"{'=' * 60}")
print(f"  Pure backbone error vs Bias: r = {corr_db:.4f}")
print(f"  SceneBias version error vs Bias: r = {corr_fb:.4f}")
if corr_db < -0.2:
    print(f"  → SceneBias compensates where backbone struggles (complementary)")
elif corr_db > 0.2:
    print(f"  → SceneBias reinforces backbone behavior")
else:
    print(f"  → SceneBias operates independently from backbone")

# ============================================================
# 可视化
# ============================================================
def tonemap(img):
    img = np.clip(img, 0, None)
    m = img.max()
    if m > 0:
        img = img / m
    img = img ** 0.45
    return (img * 255).astype(np.uint8)

img_gt_np = read_exr_rgb(str(base_dir / 'dataset/sponza_test/center/ground_truth.exr'))

gt_tm = tonemap(img_gt_np).transpose(1, 2, 0)
d_tm = tonemap(pred_d_center).transpose(1, 2, 0)
f_tm = tonemap(pred_f_center).transpose(1, 2, 0)

# ============================================================
# 图7-1 + 图7-2：降噪效果对比 + 偏置热力图
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(gt_tm)
axes[0].set_title('Ground Truth', fontsize=12, weight='bold')
axes[0].axis('off')

axes[1].imshow(d_tm)
axes[1].set_title(f'Pure TinyDenoiser\n(35.94 dB)', fontsize=11, weight='bold')
axes[1].axis('off')

axes[2].imshow(f_tm)
axes[2].set_title(f'+ SceneBias\n(35.90 dB)', fontsize=11, weight='bold')
axes[2].axis('off')

vmax = max(abs(bias_center.min()), abs(bias_center.max()))
im = axes[3].imshow(bias_center, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes[3].set_title(f'SceneBias Map\n(std={bias_center.std():.3f}, r={corr_db:.3f})', fontsize=11, weight='bold')
axes[3].axis('off')
plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(str(base_dir / 'ablation_comparison.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nablation_comparison.png saved")

# ============================================================
# 图7-3：偏置与误差空间关系
# ============================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

# 主干误差图
axes2[0].imshow(error_d_center, cmap='hot')
axes2[0].set_title('Backbone Error Map', fontsize=12, weight='bold')
axes2[0].axis('off')

# SceneBias偏置图
im2 = axes2[1].imshow(bias_center, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
axes2[1].set_title(f'SceneBias Map (r={corr_db:.3f})', fontsize=12, weight='bold')
axes2[1].axis('off')
plt.colorbar(im2, ax=axes2[1], fraction=0.046, pad=0.04)

# 散点图：误差 vs bias
axes2[2].scatter(error_d_center.flatten()[::100], bias_center.flatten()[::100],
                 alpha=0.3, s=1, c='#1565C0')
axes2[2].set_xlabel('Backbone Error', fontsize=11)
axes2[2].set_ylabel('SceneBias', fontsize=11)
axes2[2].set_title(f'Error vs Bias (r={corr_db:.3f})', fontsize=12, weight='bold')
axes2[2].axhline(y=0, color='#999', linestyle='--', linewidth=0.5)
axes2[2].axvline(x=0, color='#999', linestyle='--', linewidth=0.5)
axes2[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(str(base_dir / 'error_bias_spatial.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("error_bias_spatial.png saved")