# run_bilateral.py - 双边滤波 + 推理时间 + 内存
import cv2
import numpy as np
from pathlib import Path
from dataset.utils import read_exr_rgb, write_exr_rgb
import argparse
import time
import psutil
import os


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def bilateral_denoise(noisy_path, output_path, d=5, sigma_color=0.1, sigma_space=10):
    noisy = read_exr_rgb(str(noisy_path))
    img = noisy.transpose(1, 2, 0).copy()

    mem_before = get_memory_usage()
    start = time.time()
    result = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    elapsed = (time.time() - start) * 1000
    mem_after = get_memory_usage()

    result = result.transpose(2, 0, 1).copy()
    write_exr_rgb(str(output_path), result)

    return elapsed, mem_after - mem_before


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='dataset/sponza_test')
    parser.add_argument('--output', type=str, default='denoised_bilateral_sponza')
    parser.add_argument('--spp', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64])
    args = parser.parse_args()

    print("=" * 50)
    print("双边滤波降噪对比")
    print("=" * 50)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    views = [d.name for d in input_dir.iterdir() if d.is_dir()]

    params = {
        1:  (5, 0.3, 10), 2:  (5, 0.2, 10), 4:  (5, 0.1, 10),
        8:  (5, 0.08, 8), 16: (5, 0.05, 8), 32: (5, 0.03, 6),
        64: (5, 0.02, 6),
    }

    total = 0
    times = []

    for view in views:
        view_out = output_dir / view
        view_out.mkdir(parents=True, exist_ok=True)
        for spp in args.spp:
            noisy_path = input_dir / view / f"spp_{spp:03d}.exr"
            output_path = view_out / f"denoised_spp_{spp:03d}.exr"
            if noisy_path.exists():
                print(f"  {view} spp={spp:3d}...", end=" ")
                try:
                    d, sc, ss = params[spp]
                    t, mem = bilateral_denoise(noisy_path, output_path, d, sc, ss)
                    times.append(t)
                    print(f"✓ ({t:.1f}ms, {mem:.1f}MB)")
                    total += 1
                except Exception as e:
                    print(f"✗ ({e})")

    avg_time = np.mean(times) if times else 0
    print(f"\n双边滤波完成，共 {total} 张")
    print(f"平均推理时间: {avg_time:.1f}ms")
    print(f"参数量: 0（传统算法）")
    print(f"FLOPs: 未计算（CPU算法）")


if __name__ == '__main__':
    main()