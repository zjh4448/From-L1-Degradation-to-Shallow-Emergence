import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'

def draw_box(ax, x, y, w, h, text, color, fontsize=8, bold=False, edge_color='#444', alpha=1.0):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor=edge_color, linewidth=1.3, alpha=alpha)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            weight=weight, color='#222')

def draw_arrow(ax, x1, y1, x2, y2, color='#555', lw=1.8, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

def draw_tag(ax, x, y, text, color='#E53935'):
    ax.text(x, y, text, fontsize=8, weight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor=color, linewidth=1.2, alpha=0.95))

# ============================================================
# 图1：纯TinyDenoiser
# ============================================================
fig1, ax1 = plt.subplots(1, 1, figsize=(15, 7))
fig1.patch.set_facecolor('#FAFAFA')
ax1.set_xlim(0, 15)
ax1.set_ylim(0, 7)
ax1.axis('off')
ax1.set_facecolor('#FAFAFA')

# 标题
ax1.text(7.5, 6.7, 'TinyDenoiser Baseline Architecture (90.7K params)', ha='center', fontsize=14, weight='bold', color='#222')

# 输入
draw_box(ax1, 0.3, 2.8, 2.2, 1.5, '4 spp Input\n$I_4$ (3ch, HDR)', '#E8F5E9', fontsize=9, bold=True, edge_color='#2E7D32')

# U-Net 编码器
draw_box(ax1, 3.2, 1.5, 5.0, 4.5, '', '#E3F2FD', edge_color='#1565C0', alpha=0.6)
ax1.text(5.7, 5.6, 'TinyDenoiser U-Net', ha='center', fontsize=11, weight='bold', color='#1565C0')

# enc
draw_box(ax1, 3.6, 4.5, 4.2, 0.8, 'Encoder: enc1(3->32) -> enc2(32->48) -> enc3(48->64)\n(MaxPool downsampling + TinyBlock x3)', '#FFFFFF', fontsize=7, edge_color='#1565C0')
draw_box(ax1, 3.6, 3.5, 4.2, 0.7, 'Bottleneck: TinyBlock(64->64)', '#FFFFFF', fontsize=7, edge_color='#1565C0')
draw_box(ax1, 3.6, 2.5, 4.2, 0.8, 'Decoder: dec3(64->48) -> dec2(48->32) -> dec1(32->32)\n(TransposedConv upsampling + Skip Connections)', '#FFFFFF', fontsize=7, edge_color='#1565C0')
draw_box(ax1, 4.5, 1.8, 2.2, 0.5, 'TinyBlock: Depthwise Sep Conv + Residual', '#BBDEFB', fontsize=6, edge_color='#1565C0')

# 输出残差
draw_box(ax1, 8.8, 4.0, 2.0, 0.8, 'Noise Residual\n$R(I_4)$ (3ch)', '#FFF9C4', fontsize=8, edge_color='#F9A825')
draw_box(ax1, 11.2, 2.8, 2.5, 1.5, 'Denoised Output\n$\hat{I} = I_4 + R(I_4)$', '#C8E6C9', fontsize=9, bold=True, edge_color='#2E7D32')

# 箭头
draw_arrow(ax1, 2.5, 3.6, 3.2, 3.6, color='#2E7D32')
draw_arrow(ax1, 8.2, 4.4, 8.8, 4.4, color='#555')
draw_arrow(ax1, 10.8, 4.4, 11.2, 3.8, color='#555')
# skip
draw_arrow(ax1, 2.5, 3.0, 7.0, 1.2, color='#888', style='-')
draw_arrow(ax1, 7.0, 1.2, 12.0, 1.2, color='#888', style='-')
draw_arrow(ax1, 12.0, 1.2, 12.0, 2.8, color='#888', style='-')
ax1.text(5.5, 0.9, 'Global Residual Connection ($I_4$ bypass)', fontsize=7, color='#888', ha='center')

# 损失
draw_box(ax1, 8.8, 1.5, 2.0, 1.2, 'Combined Loss:\nChannel-W Multi-Scale L1\n+ Sobel Gradient\n+ Positive Bias', '#FFCCBC', fontsize=6.5, edge_color='#E64A19')
draw_arrow(ax1, 10.8, 2.1, 12.0, 2.5, color='#E64A19', style='-')

# 训练标注
draw_tag(ax1, 0.3, 6.4, 'Training: 4 spp in, 32 spp target', '#1565C0')

plt.tight_layout(pad=0.5)
plt.savefig('arch_tinydenoiser.png', dpi=250, bbox_inches='tight', facecolor='white')
plt.close()
print("arch_tinydenoiser.png saved")

# ============================================================
# 图2：TinyDenoiser + HFB
# ============================================================
fig2, ax2 = plt.subplots(1, 1, figsize=(15, 8))
fig2.patch.set_facecolor('#FAFAFA')
ax2.set_xlim(0, 15)
ax2.set_ylim(0, 8)
ax2.axis('off')
ax2.set_facecolor('#FAFAFA')

ax2.text(7.5, 7.7, 'TinyDenoiser + HFB Architecture (97.7K total params)', ha='center', fontsize=14, weight='bold', color='#222')

# 输入
draw_box(ax2, 0.3, 3.5, 2.2, 1.5, '4 spp Input\n$I_4$ (3ch, HDR)', '#E8F5E9', fontsize=9, bold=True, edge_color='#2E7D32')

# TinyDenoiser
draw_box(ax2, 3.0, 3.2, 2.8, 2.2, 'TinyDenoiser\nU-Net (90.7K)\n\nenc1->enc2->enc3\n->dec3->dec2->dec1', '#E3F2FD', fontsize=8, bold=True, edge_color='#1565C0')
draw_box(ax2, 3.5, 3.4, 1.8, 0.4, 'Global Residual', '#FFFFFF', fontsize=6, edge_color='#1565C0')

# HFB 旁路
draw_box(ax2, 3.0, 0.5, 5.5, 2.3, '', '#FFF3E0', edge_color='#E64A19', alpha=0.5)
ax2.text(5.75, 2.5, 'HFB (7K params, fully decoupled)', ha='center', fontsize=10, weight='bold', color='#E64A19')

draw_box(ax2, 3.3, 1.3, 1.6, 1.2, 'Gaussian\nHigh-Pass\nFilter', '#FFCCBC', fontsize=7, bold=True, edge_color='#E64A19')
draw_arrow(ax2, 4.9, 1.9, 5.5, 1.9, color='#E64A19')
draw_box(ax2, 5.5, 1.3, 2.5, 1.2, 'encoder_A\n(3->16->16)\n+ shared\n(32->16->16->16)\n+ clean_head\n(16->3, Tanh)', '#FFCCBC', fontsize=6.5, edge_color='#E64A19')

draw_box(ax2, 8.5, 4.8, 2.2, 0.8, '$H_{clean}$\n(High-Freq Compensation)', '#FFF9C4', fontsize=8, edge_color='#F9A825')
draw_box(ax2, 11.2, 3.5, 2.5, 1.8, 'Final Output\n$\hat{I} = \widetilde{I} + H_{clean}$', '#C8E6C9', fontsize=9, bold=True, edge_color='#2E7D32')

# 箭头
draw_arrow(ax2, 2.5, 4.2, 3.0, 4.2, color='#2E7D32')
draw_arrow(ax2, 2.5, 3.8, 3.3, 1.9, color='#E64A19')
draw_arrow(ax2, 5.8, 4.2, 8.5, 5.2, color='#555')
draw_arrow(ax2, 8.0, 1.9, 8.5, 4.8, color='#555')
draw_arrow(ax2, 10.7, 5.2, 11.2, 4.6, color='#555')

draw_tag(ax2, 0.3, 7.4, 'Parameters fully decoupled from backbone', '#E64A19')

plt.tight_layout(pad=0.5)
plt.savefig('arch_hfb.png', dpi=250, bbox_inches='tight', facecolor='white')
plt.close()
print("arch_hfb.png saved")

# ============================================================
# 图3：SceneBias 分阶段训练
# ============================================================
fig3, ax3 = plt.subplots(1, 1, figsize=(17, 10))
fig3.patch.set_facecolor('#FAFAFA')
ax3.set_xlim(0, 17)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_facecolor('#FAFAFA')

ax3.text(8.5, 9.7, 'SceneBias: Two-Phase Training + Inference (96.5K total params)', ha='center', fontsize=14, weight='bold', color='#222')

# ===== 阶段一 =====
ax3.add_patch(plt.Rectangle((0.3, 6.3), 8.2, 3.2, fill=False, edgecolor='#1565C0', linewidth=2.5, linestyle='-'))
draw_tag(ax3, 0.5, 9.2, 'Phase 1 (Epoch 1-30): Train Backbone, Freeze SceneBias', '#1565C0')

draw_box(ax3, 0.6, 6.8, 2.0, 1.5, '4 spp Input\n$I_4$ (3ch)', '#E8F5E9', fontsize=9, bold=True, edge_color='#2E7D32')
draw_box(ax3, 3.2, 6.8, 2.8, 1.5, 'TinyDenoiser\nU-Net (90.7K)\nTrainable', '#BBDEFB', fontsize=9, bold=True, edge_color='#1565C0')
draw_box(ax3, 6.5, 6.8, 1.8, 1.5, 'SceneBias\n(5.8K)\nFrozen\nbias = 0', '#EEEEEE', fontsize=8, edge_color='#999')

draw_arrow(ax3, 2.6, 7.5, 3.2, 7.5, color='#1565C0')
draw_arrow(ax3, 6.0, 7.5, 6.5, 7.5, color='#999', style='-')

ax3.text(4.2, 6.5, '$w=1$ (full denoise, no SceneBias effect)', fontsize=7.5, color='#888', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='#888', alpha=0.7))

# ===== 阶段二 =====
ax3.add_patch(plt.Rectangle((0.3, 2.3), 8.2, 3.2, fill=False, edgecolor='#E64A19', linewidth=2.5, linestyle='-'))
draw_tag(ax3, 0.5, 5.2, 'Phase 2 (Epoch 31-60): Freeze Backbone, Train SceneBias', '#E64A19')

draw_box(ax3, 0.6, 2.8, 2.0, 1.5, '4 spp Input\n$I_4$ (3ch)', '#E8F5E9', fontsize=9, bold=True, edge_color='#2E7D32')
draw_box(ax3, 3.2, 2.8, 2.8, 1.5, 'TinyDenoiser\nU-Net (90.7K)\nFrozen', '#E3F2FD', fontsize=9, bold=True, edge_color='#1565C0')
draw_box(ax3, 6.5, 2.8, 1.8, 1.5, 'SceneBias\n(5.8K)\nTrainable', '#FFCCBC', fontsize=8, bold=True, edge_color='#E64A19')

draw_arrow(ax3, 2.6, 3.5, 3.2, 3.5, color='#555')
draw_arrow(ax3, 6.0, 3.5, 6.5, 3.5, color='#E64A19')

# enc1通路
draw_arrow(ax3, 4.0, 2.8, 4.0, 2.1, color='#E64A19', style='-')
draw_arrow(ax3, 4.0, 2.1, 7.5, 2.1, color='#E64A19', style='-')
draw_arrow(ax3, 7.5, 2.1, 7.5, 2.8, color='#E64A19', style='-')
ax3.text(5.7, 1.8, 'enc1 feature (32ch)', fontsize=7.5, color='#E64A19', weight='bold')

# SceneBias内部结构
draw_box(ax3, 6.5, 1.2, 1.8, 0.7, 'Conv(32->16->8->1)\n3-layer, 5.8K params', '#FFF3E0', fontsize=5.5, edge_color='#E64A19')

# ===== 推理 =====
ax3.add_patch(plt.Rectangle((9.5, 2.3), 7.0, 7.0, fill=False, edgecolor='#2E7D32', linewidth=2.5, linestyle='-'))
draw_tag(ax3, 9.7, 9.2, 'Inference (Single Forward Pass)', '#2E7D32')

draw_box(ax3, 9.8, 7.0, 2.0, 1.5, '4 spp Input\n$I_4$ (3ch)', '#E8F5E9', fontsize=9, bold=True, edge_color='#2E7D32')
draw_box(ax3, 12.2, 7.0, 2.5, 1.5, 'TinyDenoiser\n$\widetilde{I}$ (Initial)', '#BBDEFB', fontsize=9, edge_color='#1565C0')

draw_arrow(ax3, 11.8, 7.7, 12.2, 7.7, color='#2E7D32')

# enc1 + SceneBias
draw_box(ax3, 12.2, 5.2, 2.5, 1.2, 'enc1 (32ch)\nShallow Features', '#90CAF9', fontsize=7, edge_color='#1565C0')
draw_arrow(ax3, 13.4, 6.8, 13.4, 6.4, color='#E64A19', style='-')

draw_box(ax3, 15.2, 5.2, 1.3, 1.2, 'SceneBias\nbias', '#FFCCBC', fontsize=7, bold=True, edge_color='#E64A19')
draw_arrow(ax3, 14.7, 6.0, 15.2, 5.8, color='#E64A19', style='-')

# 调制
draw_box(ax3, 9.8, 3.3, 6.7, 1.5, '', '#C8E6C9', edge_color='#2E7D32')
ax3.text(13.15, 4.5, 'Spatial-Adaptive Modulation', ha='center', fontsize=9, weight='bold', color='#222')
ax3.text(13.15, 4.0, '$w = \\mathrm{clamp}(1 - \\mathrm{bias},\\, 0,\\, 1)$', ha='center', fontsize=8, color='#333')
ax3.text(13.15, 3.6, '$\hat{I} = I_4 - w \\cdot (I_4 - \\widetilde{I})$', ha='center', fontsize=9, weight='bold', color='#222')

draw_arrow(ax3, 13.4, 5.2, 13.4, 4.8, color='#555')
draw_arrow(ax3, 16.0, 5.5, 16.0, 4.5, color='#555')

# 关键说明
ax3.text(9.8, 2.5, 'Pure RGB input | No Albedo/Normal/Depth | Zero extra IO', fontsize=7.5, color='#2E7D32', weight='bold')

plt.tight_layout(pad=0.5)
plt.savefig('arch_scenebias_training.png', dpi=250, bbox_inches='tight', facecolor='white')
plt.close()
print("arch_scenebias_training.png saved")