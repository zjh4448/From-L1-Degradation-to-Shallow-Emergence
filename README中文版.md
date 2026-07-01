
# TinyDenoiser + SceneBias — 实验指令


## 环境配置

### 创建虚拟环境并安装依赖

```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> **说明**：`requirements.txt` 为完整环境导出，但**不包含** `torch`、`torchvision`、`torchaudio`，需通过上方 PyTorch 安装命令单独安装，以避免版本冲突。

---

## 一、ABCD消融实验（论文第3章）
使用Cornell Box数据集进行训练
> 对应论文 Table 1（定量消融结果）及 Table 2（多采样率对比）。

### 1.1 训练

```bash
python train.py --config config_A.yaml    # 纯L1
python train.py --config config_B.yaml    # +多尺度损失
python train.py --config config_C.yaml    # +梯度损失
python train.py --config config_D.yaml    # +通道加权 + 亮度惩罚（完整复合损失）
```

### 1.2 推理

```bash
python inference.py --model output_n2n/A/checkpoints/best_model.pth --config config_A.yaml --input dataset/test --output denoised_A --batch
python inference.py --model output_n2n/B/checkpoints/best_model.pth --config config_B.yaml --input dataset/test --output denoised_B --batch
python inference.py --model output_n2n/C/checkpoints/best_model.pth --config config_C.yaml --input dataset/test --output denoised_C --batch
python inference.py --model output_n2n/D/checkpoints/best_model.pth --config config_D.yaml --input dataset/test --output denoised_D --batch
```

### 1.3 评估

```bash
python evaluate.py --denoised_dir denoised_A --gt_dir dataset/test --baseline
python evaluate.py --denoised_dir denoised_B --gt_dir dataset/test --baseline
python evaluate.py --denoised_dir denoised_C --gt_dir dataset/test --baseline
python evaluate.py --denoised_dir denoised_D --gt_dir dataset/test --baseline
```

### 1.4 基准测试（FLOPs / 显存 / 推理速度）

```bash
python inference.py --model output_n2n/D/checkpoints/best_model.pth --config config_D.yaml --input dataset/test/front/spp_004.exr --benchmark
```

### 1.5 Sponza 零样本泛化

> 对应论文 Table 5（零样本跨场景泛化测试）。

```bash
python inference.py --model output_n2n/D/checkpoints/best_model.pth --config config_D.yaml --input dataset/sponza_test --output denoised_sponza --batch
python evaluate.py --denoised_dir denoised_sponza --gt_dir dataset/sponza_test --scenes center corner left right up --spp 4 --pattern "denoised_spp_004.exr" --baseline
```

---

## 二、对比方法评估（论文第7章）

> 对应论文 Table 3（同场景性能对比）及 Table 5（Sponza泛化对比）。

### 2.1 Intel OIDN

```bash
# Cornell Box
python run_oidn.py
python evaluate.py --denoised_dir denoised_oidn --gt_dir dataset/test --baseline

# Sponza
python run_oidn.py --input dataset/sponza_test --output denoised_oidn_sponza --spp 4
python evaluate.py --denoised_dir denoised_oidn_sponza --gt_dir dataset/sponza_test --scenes center corner left right up --spp 4 --baseline
```

### 2.2 双边滤波

```bash
# Cornell Box
python run_bilateral.py
python evaluate.py --denoised_dir denoised_bilateral --gt_dir dataset/test --baseline

# Sponza
python run_bilateral.py --input dataset/sponza_test --output denoised_bilateral_sponza --spp 4
python evaluate.py --denoised_dir denoised_bilateral_sponza --gt_dir dataset/sponza_test --scenes center corner left right up --spp 4 --baseline
```

---

## 三、SceneBias 空间偏置学习（论文第6章）
使用Sponza数据集训练
> 对应论文 Table 4（SceneBias有效性验证）、图6-1（架构图）及图7-1/7-2（可视化）。

### 3.1 分阶段训练

```bash
python train.py --config config_E.yaml
```

| 阶段 | Epoch | 训练对象 | SceneBias状态 |
|------|-------|---------|---------------|
| 阶段一 | 1–30 | TinyDenoiser主干 | 冻结（bias=0, w=1） |
| 阶段二 | 31–60 | SceneBias模块 | 主干冻结，学习空间偏置 |

### 3.2 推理与评估

```bash
# 推理
python inference.py --model output_n2n/F/checkpoints/best_model.pth --config config_E.yaml --input dataset/sponza_test --output denoised_F_sponza --batch

# 评估（PSNR / SSIM）
python evaluate.py --denoised_dir denoised_F_sponza --gt_dir dataset/sponza_test --scenes center corner left right up --spp 4 --pattern "denoised_spp_004.exr"

# 感知指标（FLIP）
python FLIP.py --denoised_dir denoised_F_sponza --gt_dir dataset/sponza_test --scenes center corner left right up --spp 4 --pattern "denoised_spp_004.exr" --baseline
```

### 3.3 SceneBias 消融对比

```bash
python scenebias_ablation.py
```

**输出内容**：
- 5个视角的线性域PSNR对比（纯主干 vs +SceneBias）
- SceneBias偏置的空间统计（均值/标准差/min/max）
- 偏置与主干误差的Pearson相关系数（论文值：`r = -0.257`）
- 生成论文图表：
  - `ablation_comparison.png`（图7-1：GT / 纯主干 / +SceneBias / 偏置热力图）
  - `error_bias_spatial.png`（图7-2：误差图 / 偏置图 / 误差-偏置散点图）

### 3.4 纯主干对比基线（Sponza同场景，30轮）

```bash
python train.py --config config_D_sponza.yaml
python inference.py --model output_n2n/D_sponza/checkpoints/best_model.pth --config config_D_sponza.yaml --input dataset/sponza_test --output denoised_D_sponza --batch
python evaluate.py --denoised_dir denoised_D_sponza --gt_dir dataset/sponza_test --scenes center corner left right up --spp 4 --pattern "denoised_spp_004.exr"
python FLIP.py --denoised_dir denoised_D_sponza --gt_dir dataset/sponza_test --scenes center corner left right up --spp 4 --pattern "denoised_spp_004.exr" --baseline
```

---

## 四、可视化

### 降噪效果对比图

```bash
# Cornell Box
python duibi.py --noisy dataset/test/front/spp_004.exr --denoised denoised_D/front/denoised_spp_004.exr --gt dataset/test/front/ground_truth.exr --output comparison_front.png

# Sponza
python duibi.py --noisy dataset/sponza_test/center/spp_004.exr --denoised denoised_F_sponza/center/denoised_spp_004.exr --gt dataset/sponza_test/center/ground_truth.exr --output comparison_sponza.png
```

---

## 五、模型导出

```bash
python export_torchscript.py
```

---

## 目录结构

```
ai_w/
├── models/
│   ├── unet.py              # TinyDenoiser + SceneBias
│   ├── scene_bias.py        # SceneBias模块
│   └── losses.py            # L1损失
├── dataset/
│   ├── dataset.py
│   ├── transforms.py
│   ├── utils.py
│   ├── train/               # 训练数据（Cornell Box）
│   ├── test/                # 测试数据（Cornell Box，5视角）
│   └── sponza_test/         # Sponza测试集（5视角）
├── output_n2n/
│   ├── A/                   # Config A checkpoint
│   ├── B/                   # Config B checkpoint
│   ├── C/                   # Config C checkpoint
│   ├── D/                   # Config D checkpoint（Cornell Box）
│   ├── D_sponza/            # Config D Sponza同场景
│   └── F/                   # Config E SceneBias分阶段训练
├── config_A.yaml
├── config_B.yaml
├── config_C.yaml
├── config_D.yaml
├── config_D_sponza.yaml
├── config_E.yaml
├── train.py
├── inference.py
├── evaluate.py
├── FLIP.py
├── scenebias_ablation.py
├── run_bilateral.py
├── run_oidn.py
├── duibi.py
├── export_torchscript.py
└── requirements.txt
```

---

