# evaluate.py - 添加FLIP指标评估
import numpy as np
from pathlib import Path
import argparse
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from dataset.utils import read_exr_rgb

# FLIP评估
try:
    from flip_evaluator.flip_python_api import evaluate as flip_evaluate

    HAS_FLIP = True
except ImportError:
    HAS_FLIP = False
    print("⚠️ flip_evaluator未安装，FLIP指标将跳过。安装: pip install flip-evaluator")


def _compute_data_range(gt):
    dr = gt.max() - gt.min()
    return dr if dr > 0 else 1.0


def compute_flip_score(denoised, gt):
    """计算FLIP分数（值越小越好）"""
    if not HAS_FLIP:
        return None

    denoised_hwc = denoised.transpose(1, 2, 0).astype(np.float32)
    gt_hwc = gt.transpose(1, 2, 0).astype(np.float32)

    _, mean_flip, _ = flip_evaluate(gt_hwc, denoised_hwc, "HDR", computeMeanError=True)
    return float(mean_flip)


def evaluate_scene(denoised_dir, gt_dir, scenes, spp_list=None, denoised_pattern="denoised_{input_name}"):
    if spp_list is None:
        spp_list = [4]

    denoised_dir = Path(denoised_dir)
    gt_dir = Path(gt_dir)
    results = {}

    for spp in spp_list:
        print(f"\n{'=' * 60}")
        print(f"Evaluating spp={spp}")
        print(f"{'=' * 60}")

        psnr_list = []
        ssim_list = []
        flip_list = []
        scene_results = {}

        input_name = f"spp_{spp:03d}.exr"
        denoised_name = denoised_pattern.format(input_name=input_name, spp=spp)

        for scene in scenes:
            denoised_path = denoised_dir / scene / denoised_name
            gt_path = gt_dir / scene / "ground_truth.exr"

            if not denoised_path.exists():
                print(f"  {scene}: SKIPPED (file not found)")
                continue
            if not gt_path.exists():
                print(f"  {scene}: SKIPPED (GT not found)")
                continue

            try:
                denoised = read_exr_rgb(str(denoised_path))
                gt = read_exr_rgb(str(gt_path))

                min_h = min(denoised.shape[1], gt.shape[1])
                min_w = min(denoised.shape[2], gt.shape[2])
                denoised = denoised[:, :min_h, :min_w]
                gt = gt[:, :min_h, :min_w]

                data_range = _compute_data_range(gt)

                psnr = peak_signal_noise_ratio(gt, denoised, data_range=data_range)
                ssim = structural_similarity(
                    gt.transpose(1, 2, 0),
                    denoised.transpose(1, 2, 0),
                    channel_axis=2,
                    data_range=data_range
                )

                flip = compute_flip_score(denoised, gt)

                psnr_list.append(psnr)
                ssim_list.append(ssim)
                if flip is not None:
                    flip_list.append(flip)

                flip_str = f", FLIP={flip:.4f}" if flip is not None else ""
                print(f"  {scene}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}{flip_str}")

            except Exception as e:
                print(f"  {scene}: ERROR ({e})")
                continue

        if psnr_list:
            avg_psnr = np.mean(psnr_list)
            avg_ssim = np.mean(ssim_list)
            std_psnr = np.std(psnr_list)
            std_ssim = np.std(ssim_list)

            print(f"  {'─' * 50}")
            print(f"  Avg PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
            print(f"  Avg SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
            if flip_list:
                print(f"  Avg FLIP: {np.mean(flip_list):.4f} ± {np.std(flip_list):.4f}")

            results[spp] = {
                'avg_psnr': round(avg_psnr, 2),
                'std_psnr': round(std_psnr, 2),
                'avg_ssim': round(avg_ssim, 4),
                'std_ssim': round(std_ssim, 4),
                'avg_flip': round(np.mean(flip_list), 4) if flip_list else None,
                'std_flip': round(np.std(flip_list), 4) if flip_list else None,
                'scenes': scene_results
            }
        else:
            results[spp] = None

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    header = f"{'SPP':<8} {'PSNR (dB)':<18} {'SSIM':<14} {'FLIP':<12} {'Scenes':<8}"
    print(header)
    print(f"{'─' * 60}")

    for spp, res in results.items():
        if res:
            n_scenes = len(res['scenes'])
            flip_str = f"{res['avg_flip']:.4f}" if res.get('avg_flip') is not None else "N/A"
            print(f"  {spp:<6} {res['avg_psnr']:.2f} ± {res['std_psnr']:.2f}     "
                  f"{res['avg_ssim']:.4f} ± {res['std_ssim']:.4f}   "
                  f"{flip_str:<12} {n_scenes}/{len(scenes)}")

    return results


def evaluate_baseline(gt_dir, scenes, spp_list=None):
    if spp_list is None:
        spp_list = [4]

    gt_dir = Path(gt_dir)
    print(f"\n{'=' * 60}")
    print(f"BASELINE: Noisy vs Ground Truth")
    print(f"{'=' * 60}")

    baseline_results = {}

    for spp in spp_list:
        psnr_list = []
        ssim_list = []
        flip_list = []

        for scene in scenes:
            noisy_path = gt_dir / scene / f"spp_{spp:03d}.exr"
            gt_path = gt_dir / scene / "ground_truth.exr"

            if not noisy_path.exists() or not gt_path.exists():
                continue

            try:
                noisy = read_exr_rgb(str(noisy_path))
                gt = read_exr_rgb(str(gt_path))

                min_h = min(noisy.shape[1], gt.shape[1])
                min_w = min(noisy.shape[2], gt.shape[2])
                noisy = noisy[:, :min_h, :min_w]
                gt = gt[:, :min_h, :min_w]

                data_range = _compute_data_range(gt)

                psnr = peak_signal_noise_ratio(gt, noisy, data_range=data_range)
                ssim = structural_similarity(
                    gt.transpose(1, 2, 0), noisy.transpose(1, 2, 0),
                    channel_axis=2, data_range=data_range
                )
                flip = compute_flip_score(noisy, gt)

                psnr_list.append(psnr)
                ssim_list.append(ssim)
                if flip is not None:
                    flip_list.append(flip)

            except Exception:
                continue

        if psnr_list:
            baseline_results[spp] = {
                'avg_psnr': round(np.mean(psnr_list), 2),
                'avg_ssim': round(np.mean(ssim_list), 4),
                'avg_flip': round(np.mean(flip_list), 4) if flip_list else None,
                'count': len(psnr_list)
            }

    print(f"\n{'SPP':<8} {'PSNR (dB)':<14} {'SSIM':<12} {'FLIP':<10} {'Samples':<8}")
    print(f"{'─' * 50}")
    for spp, res in baseline_results.items():
        flip_str = f"{res['avg_flip']:.4f}" if res.get('avg_flip') is not None else "N/A"
        print(f"  {spp:<6} {res['avg_psnr']:.2f}          {res['avg_ssim']:.4f}      {flip_str:<10} {res['count']}")

    return baseline_results


def main():
    parser = argparse.ArgumentParser(description='评估降噪结果（PSNR/SSIM/FLIP）')
    parser.add_argument('--denoised_dir', type=str, default='denoised_results')
    parser.add_argument('--gt_dir', type=str, default='dataset/test')
    parser.add_argument('--scenes', type=str, nargs='+',
                        default=['front', 'right_high', 'left_low', 'far', 'angled'])
    parser.add_argument('--spp', type=int, nargs='+', default=[4])
    parser.add_argument('--pattern', type=str, default='denoised_{input_name}')
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()

    evaluate_scene(args.denoised_dir, args.gt_dir, args.scenes, args.spp, args.pattern)
    if args.baseline:
        evaluate_baseline(args.gt_dir, args.scenes, args.spp)


if __name__ == '__main__':
    main()