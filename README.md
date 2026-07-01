
# TinyDenoiser + SceneBias — Experiment Instructions


## Environment Setup

### Create Virtual Environment and Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

> **Note**: `requirements.txt` contains the full environment export but **excludes** `torch`, `torchvision`, and `torchaudio` to avoid version conflicts. These must be installed separately via the PyTorch command above.
**Translation:**

> **Note**: The data generation scripts are available on the [Releases](https://github.com/zjh4448/From-L1-Degradation-to-Shallow-Emergence/releases) page of this repository. Please download `ai_py.zip` separately to generate the data. Place the generated `test`, `train`, `val`, and `sponza` folders under the `dataset` directory. It is worth noting that there are two different versions of the `train` and `val` datasets; please choose the appropriate one according to the instructions below.

---


## I. ABCD Ablation Experiments (Chapter 3)
Trained on the Cornell Box dataset.
> Corresponds to Table 1 (quantitative ablation results) and Table 2 (multi-spp comparison) in the paper.

### 1.1 Training

```bash
python train.py --config config_A.yaml    # Pure L1
python train.py --config config_B.yaml    # +Multi-scale loss
python train.py --config config_C.yaml    # +Gradient loss
python train.py --config config_D.yaml    # +Channel weighting + Brightness penalty (full composite loss)
```

### 1.2 Inference

```bash
python inference.py --model output_n2n/A/checkpoints/best_model.pth --config config_A.yaml --input dataset/test --output denoised_A --batch
python inference.py --model output_n2n/B/checkpoints/best_model.pth --config config_B.yaml --input dataset/test --output denoised_B --batch
python inference.py --model output_n2n/C/checkpoints/best_model.pth --config config_C.yaml --input dataset/test --output denoised_C --batch
python inference.py --model output_n2n/D/checkpoints/best_model.pth --config config_D.yaml --input dataset/test --output denoised_D --batch
```

### 1.3 Evaluation

```bash
python evaluate.py --denoised_dir denoised_A --gt_dir dataset/test --baseline
python evaluate.py --denoised_dir denoised_B --gt_dir dataset/test --baseline
python evaluate.py --denoised_dir denoised_C --gt_dir dataset/test --baseline
python evaluate.py --denoised_dir denoised_D --gt_dir dataset/test --baseline
```

### 1.4 Benchmark (FLOPs / Memory / Inference Speed)

```bash
python inference.py --model output_n2n/D/checkpoints/best_model.pth --config config_D.yaml --input dataset/test/front/spp_004.exr --benchmark
```

### 1.5 Sponza Zero-Shot Generalization

> Corresponds to Table 5 (zero-shot cross-scene generalization) in the paper.

```bash
python inference.py --model output_n2n/D/checkpoints/best_model.pth --config config_D.yaml --input dataset/sponza_test --output denoised_sponza --batch
python evaluate.py --denoised_dir denoised_sponza --gt_dir dataset/sponza_test --scenes center corner left right up --spp 4 --pattern "denoised_spp_004.exr" --baseline
```

---

## II. Baseline Comparisons (Chapter 7)

> Corresponds to Table 3 (intra-scene comparison) and Table 5 (Sponza generalization comparison) in the paper.

### 2.1 Intel OIDN

```bash
# Cornell Box
python run_oidn.py
python evaluate.py --denoised_dir denoised_oidn --gt_dir dataset/test --baseline

# Sponza
python run_oidn.py --input dataset/sponza_test --output denoised_oidn_sponza --spp 4
python evaluate.py --denoised_dir denoised_oidn_sponza --gt_dir dataset/sponza_test --scenes center corner left right up --spp 4 --baseline
```

### 2.2 Bilateral Filter

```bash
# Cornell Box
python run_bilateral.py
python evaluate.py --denoised_dir denoised_bilateral --gt_dir dataset/test --baseline

# Sponza
python run_bilateral.py --input dataset/sponza_test --output denoised_bilateral_sponza --spp 4
python evaluate.py --denoised_dir denoised_bilateral_sponza --gt_dir dataset/sponza_test --scenes center corner left right up --spp 4 --baseline
```

---

## III. SceneBias Spatial Bias Learning (Chapter 6)
Trained on the Sponza dataset.
> Corresponds to Table 4 (SceneBias effectiveness validation), Figure 6-1 (architecture), and Figures 7-1/7-2 (visualizations) in the paper.

### 3.1 Two-Stage Training

```bash
python train.py --config config_E.yaml
```

| Stage | Epochs | Training Target | SceneBias Status |
|-------|--------|-----------------|------------------|
| Stage 1 | 1–30 | TinyDenoiser backbone | Frozen (bias=0, w=1) |
| Stage 2 | 31–60 | SceneBias module | Backbone frozen, learning spatial bias |

### 3.2 Inference and Evaluation

```bash
# Inference
python inference.py --model output_n2n/F/checkpoints/best_model.pth --config config_E.yaml --input dataset/sponza_test --output denoised_F_sponza --batch

# Evaluation (PSNR / SSIM)
python evaluate.py --denoised_dir denoised_F_sponza --gt_dir dataset/sponza_test --scenes center corner left right up --spp 4 --pattern "denoised_spp_004.exr"

# Perceptual metric (FLIP)
python FLIP.py --denoised_dir denoised_F_sponza --gt_dir dataset/sponza_test --scenes center corner left right up --spp 4 --pattern "denoised_spp_004.exr" --baseline
```

### 3.3 SceneBias Ablation Comparison

```bash
python scenebias_ablation.py
```

**Outputs**:
- Linear-domain PSNR comparison across 5 views (pure backbone vs +SceneBias)
- Spatial statistics of SceneBias bias maps (mean / std / min / max)
- Pearson correlation between bias and backbone reconstruction error (paper value: `r = -0.257`)
- Generates paper figures:
  - `ablation_comparison.png` (Figure 7-1: GT / Pure backbone / +SceneBias / Bias heatmap)
  - `error_bias_spatial.png` (Figure 7-2: Error map / Bias map / Error-Bias scatter plot)

### 3.4 Pure Backbone Baseline (Sponza same-scene, 30 epochs)

```bash
python train.py --config config_D_sponza.yaml
python inference.py --model output_n2n/D_sponza/checkpoints/best_model.pth --config config_D_sponza.yaml --input dataset/sponza_test --output denoised_D_sponza --batch
python evaluate.py --denoised_dir denoised_D_sponza --gt_dir dataset/sponza_test --scenes center corner left right up --spp 4 --pattern "denoised_spp_004.exr"
python FLIP.py --denoised_dir denoised_D_sponza --gt_dir dataset/sponza_test --scenes center corner left right up --spp 4 --pattern "denoised_spp_004.exr" --baseline
```

---

## IV. Visualization

### Denoising Comparison Figures

```bash
# Cornell Box
python duibi.py --noisy dataset/test/front/spp_004.exr --denoised denoised_D/front/denoised_spp_004.exr --gt dataset/test/front/ground_truth.exr --output comparison_front.png

# Sponza
python duibi.py --noisy dataset/sponza_test/center/spp_004.exr --denoised denoised_F_sponza/center/denoised_spp_004.exr --gt dataset/sponza_test/center/ground_truth.exr --output comparison_sponza.png
```

---

## V. Model Export

```bash
python export_torchscript.py
```

---

## Directory Structure

```
ai_w/
├── models/
│   ├── unet.py              # TinyDenoiser + SceneBias
│   ├── scene_bias.py        # SceneBias module
│   └── losses.py            # L1 loss
├── dataset/
│   ├── dataset.py
│   ├── transforms.py
│   ├── utils.py
│   ├── train/               # Training data (Cornell Box)
│   ├── test/                # Test data (Cornell Box, 5 views)
│   └── sponza_test/         # Sponza test set (5 views)
├── output_n2n/
│   ├── A/                   # Config A checkpoint
│   ├── B/                   # Config B checkpoint
│   ├── C/                   # Config C checkpoint
│   ├── D/                   # Config D checkpoint (Cornell Box)
│   ├── D_sponza/            # Config D Sponza same-scene
│   └── F/                   # Config E SceneBias two-stage training
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


