# evaluate.py - 修复版（统一 data_range = gt.max() - gt.min()）
import numpy as np
from pathlib import Path
import argparse
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from dataset.utils import read_exr_rgb


def _compute_data_range(gt):
    """统一计算 data_range：只用 GT 的范围，保证公平对比"""
    dr = gt.max() - gt.min()
    return dr if dr > 0 else 1.0


def evaluate_scene(denoised_dir, gt_dir, scenes, spp_list=None, denoised_pattern="denoised_{input_name}"):
    if spp_list is None:
        spp_list = [4, 64]

    denoised_dir = Path(denoised_dir)
    gt_dir = Path(gt_dir)
    results = {}

    for spp in spp_list:
        print(f"\n{'='*60}")
        print(f"Evaluating spp={spp}")
        print(f"{'='*60}")

        psnr_list = []
        ssim_list = []
        scene_results = {}

        input_name = f"spp_{spp:03d}.exr"
        denoised_name = denoised_pattern.format(input_name=input_name, spp=spp)

        for scene in scenes:
            denoised_path = denoised_dir / scene / denoised_name
            gt_path = gt_dir / scene / "ground_truth.exr"

            if not denoised_path.exists():
                print(f"  {scene}: ⚠️ SKIPPED (降噪文件不存在: {denoised_path})")
                continue
            if not gt_path.exists():
                print(f"  {scene}: ⚠️ SKIPPED (GT不存在: {gt_path})")
                continue

            try:
                denoised = read_exr_rgb(str(denoised_path))
                gt = read_exr_rgb(str(gt_path))

                # 对齐尺寸
                min_h = min(denoised.shape[1], gt.shape[1])
                min_w = min(denoised.shape[2], gt.shape[2])
                denoised = denoised[:, :min_h, :min_w]
                gt = gt[:, :min_h, :min_w]

                # 统一 data_range = GT 的范围
                data_range = _compute_data_range(gt)

                psnr = peak_signal_noise_ratio(gt, denoised, data_range=data_range)
                ssim = structural_similarity(
                    gt.transpose(1, 2, 0),
                    denoised.transpose(1, 2, 0),
                    channel_axis=2,
                    data_range=data_range
                )

                psnr_list.append(psnr)
                ssim_list.append(ssim)
                scene_results[scene] = {'psnr': round(psnr, 2), 'ssim': round(ssim, 4)}
                print(f"  {scene}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")

            except Exception as e:
                print(f"  {scene}: ❌ ERROR ({e})")
                continue

        if psnr_list:
            avg_psnr = np.mean(psnr_list)
            avg_ssim = np.mean(ssim_list)
            std_psnr = np.std(psnr_list)
            std_ssim = np.std(ssim_list)

            print(f"  {'─'*50}")
            print(f"  Avg PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
            print(f"  Avg SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")

            results[spp] = {
                'avg_psnr': round(avg_psnr, 2),
                'std_psnr': round(std_psnr, 2),
                'avg_ssim': round(avg_ssim, 4),
                'std_ssim': round(std_ssim, 4),
                'scenes': scene_results
            }
        else:
            print(f"  ❌ No valid results for spp={spp}")
            results[spp] = None

    # 汇总
    print(f"\n{'='*60}")
    print(f"SUMMARY - 降噪结果")
    print(f"{'='*60}")
    print(f"{'SPP':<8} {'PSNR (dB)':<18} {'SSIM':<14} {'Scenes':<8}")
    print(f"{'─'*48}")

    for spp, res in results.items():
        if res:
            n_scenes = len(res['scenes'])
            print(f"  {spp:<6} {res['avg_psnr']:.2f} ± {res['std_psnr']:.2f}     "
                  f"{res['avg_ssim']:.4f} ± {res['std_ssim']:.4f}   "
                  f"{n_scenes}/{len(scenes)}")

    valid_results = {k: v for k, v in results.items() if v}
    if valid_results:
        best_psnr_spp = max(valid_results, key=lambda k: valid_results[k]['avg_psnr'])
        best_ssim_spp = max(valid_results, key=lambda k: valid_results[k]['avg_ssim'])
        print(f"\n  Best PSNR: spp={best_psnr_spp} ({valid_results[best_psnr_spp]['avg_psnr']:.2f} dB)")
        print(f"  Best SSIM: spp={best_ssim_spp} ({valid_results[best_ssim_spp]['avg_ssim']:.4f})")

    return results


def evaluate_baseline(gt_dir, scenes, spp_list=None):
    """评估原始噪声图 vs GT，使用相同 data_range 保证公平对比"""
    if spp_list is None:
        spp_list = [1, 2, 4, 8, 16, 32, 64]

    gt_dir = Path(gt_dir)
    print(f"\n{'='*60}")
    print(f"BASELINE: 原始噪声图 vs Ground Truth")
    print(f"{'='*60}")

    baseline_results = {}

    for spp in spp_list:
        psnr_list = []
        ssim_list = []

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

                # 统一 data_range = GT 的范围
                data_range = _compute_data_range(gt)

                psnr = peak_signal_noise_ratio(gt, noisy, data_range=data_range)
                ssim = structural_similarity(
                    gt.transpose(1, 2, 0), noisy.transpose(1, 2, 0),
                    channel_axis=2, data_range=data_range
                )

                psnr_list.append(psnr)
                ssim_list.append(ssim)

            except Exception:
                continue

        if psnr_list:
            avg_psnr = np.mean(psnr_list)
            avg_ssim = np.mean(ssim_list)
            baseline_results[spp] = {
                'avg_psnr': round(avg_psnr, 2),
                'avg_ssim': round(avg_ssim, 4),
                'count': len(psnr_list)
            }

    print(f"\n{'SPP':<8} {'PSNR (dB)':<14} {'SSIM':<12} {'Samples':<8}")
    print(f"{'─'*42}")
    for spp, res in baseline_results.items():
        print(f"  {spp:<6} {res['avg_psnr']:.2f}          {res['avg_ssim']:.4f}      {res['count']}")

    return baseline_results


def print_comparison(denoised_results, baseline_results):
    """打印降噪 vs 原始噪声的对比"""
    if not denoised_results or not baseline_results:
        return

    print(f"\n{'='*60}")
    print(f"对比: 降噪 vs 原始噪声")
    print(f"{'='*60}")
    print(f"{'SPP':<8} {'原始 PSNR':<12} {'降噪 PSNR':<12} {'提升':<10} {'原始 SSIM':<12} {'降噪 SSIM':<12}")
    print(f"{'─'*66}")

    for spp in sorted(set(list(denoised_results.keys()) + list(baseline_results.keys()))):
        d = denoised_results.get(spp, {})
        b = baseline_results.get(spp, {})

        if d and b:
            psnr_gain = d['avg_psnr'] - b['avg_psnr']
            ssim_gain = d['avg_ssim'] - b['avg_ssim']
            sign = '+' if psnr_gain >= 0 else ''
            print(f"  {spp:<6} {b['avg_psnr']:.2f}         {d['avg_psnr']:.2f}         "
                  f"{sign}{psnr_gain:.2f}       {b['avg_ssim']:.4f}        {d['avg_ssim']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='评估降噪结果（监督学习版本）')
    parser.add_argument('--denoised_dir', type=str, default='denoised_results')
    parser.add_argument('--gt_dir', type=str, default='dataset/test')
    parser.add_argument('--scenes', type=str, nargs='+',
                        default=['front', 'right_high', 'left_low', 'far', 'angled'])
    parser.add_argument('--spp', type=int, nargs='+',
                        default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--pattern', type=str, default='denoised_{input_name}')
    parser.add_argument('--baseline', action='store_true',
                        help='同时评估原始噪声图作为baseline')
    args = parser.parse_args()

    denoised_results = evaluate_scene(
        args.denoised_dir, args.gt_dir, args.scenes, args.spp, args.pattern
    )

    baseline_results = None
    if args.baseline:
        baseline_results = evaluate_baseline(args.gt_dir, args.scenes, args.spp)
        print_comparison(denoised_results, baseline_results)


if __name__ == '__main__':
    main()