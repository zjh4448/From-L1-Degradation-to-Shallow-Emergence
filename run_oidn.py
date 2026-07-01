# run_oidn.py - 适配 NewDevice API
import oidn
import numpy as np
from pathlib import Path
from dataset.utils import read_exr_rgb, write_exr_rgb
import argparse


def oidn_denoise(noisy_path, output_path):
    noisy = read_exr_rgb(str(noisy_path))  # (3, H, W)
    _, H, W = noisy.shape
    # 转为 (H, W, 3) 连续内存
    img = noisy.transpose(1, 2, 0).copy()  # (512, 512, 3)

    device = oidn.NewDevice(oidn.DEVICE_TYPE_DEFAULT)
    oidn.CommitDevice(device)

    oidn_filter = oidn.NewFilter(device, "RT")
    oidn.SetSharedFilterImage(oidn_filter, "color", img, oidn.FORMAT_FLOAT3, W, H, 0, 0, 0)
    oidn.SetSharedFilterImage(oidn_filter, "output", img, oidn.FORMAT_FLOAT3, W, H, 0, 0, 0)
    oidn.CommitFilter(oidn_filter)
    oidn.ExecuteFilter(oidn_filter)

    # 转回 (3, H, W)
    result = img.transpose(2, 0, 1).copy()
    write_exr_rgb(str(output_path), result)

    oidn.ReleaseFilter(oidn_filter)
    oidn.ReleaseDevice(device)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='dataset/test')
    parser.add_argument('--output', type=str, default='denoised_oidn')
    parser.add_argument('--spp', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64])
    args = parser.parse_args()

    print("=" * 50)
    print("Intel OIDN 降噪对比")
    print("=" * 50)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    views = [d.name for d in input_dir.iterdir() if d.is_dir()]
    total = 0

    for view in views:
        view_out = output_dir / view
        view_out.mkdir(parents=True, exist_ok=True)
        for spp in args.spp:
            noisy_path = input_dir / view / f"spp_{spp:03d}.exr"
            output_path = view_out / f"denoised_spp_{spp:03d}.exr"
            if noisy_path.exists():
                print(f"  {view} spp={spp:3d}...", end=" ")
                try:
                    oidn_denoise(noisy_path, output_path)
                    print("✓")
                    total += 1
                except Exception as e:
                    print(f"✗ ({e})")
    print(f"\nOIDN 完成，共 {total} 张")


if __name__ == '__main__':
    main()# run_oidn.py - 加推理时间 + 内存
import oidn
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


def oidn_denoise(noisy_path, output_path):
    noisy = read_exr_rgb(str(noisy_path))
    _, H, W = noisy.shape
    img = noisy.transpose(1, 2, 0).copy()

    device = oidn.NewDevice(oidn.DEVICE_TYPE_DEFAULT)
    oidn.CommitDevice(device)

    oidn_filter = oidn.NewFilter(device, "RT")
    oidn.SetSharedFilterImage(oidn_filter, "color", img, oidn.FORMAT_FLOAT3, W, H, 0, 0, 0)
    oidn.SetSharedFilterImage(oidn_filter, "output", img, oidn.FORMAT_FLOAT3, W, H, 0, 0, 0)
    oidn.CommitFilter(oidn_filter)

    mem_before = get_memory_usage()
    start = time.time()
    oidn.ExecuteFilter(oidn_filter)
    elapsed = (time.time() - start) * 1000
    mem_after = get_memory_usage()

    result = img.transpose(2, 0, 1).copy()
    write_exr_rgb(str(output_path), result)

    oidn.ReleaseFilter(oidn_filter)
    oidn.ReleaseDevice(device)

    return elapsed, mem_after - mem_before


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='dataset/sponza_test')
    parser.add_argument('--output', type=str, default='denoised_oidn_sponza')
    parser.add_argument('--spp', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64])
    args = parser.parse_args()

    print("=" * 50)
    print("Intel OIDN 降噪对比")
    print("=" * 50)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    views = [d.name for d in input_dir.iterdir() if d.is_dir()]
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
                    t, mem = oidn_denoise(noisy_path, output_path)
                    times.append(t)
                    print(f"✓ ({t:.1f}ms, {mem:.1f}MB)")
                    total += 1
                except Exception as e:
                    print(f"✗ ({e})")

    avg_time = np.mean(times) if times else 0
    print(f"\nOIDN 完成，共 {total} 张")
    print(f"平均推理时间: {avg_time:.1f}ms")
    print(f"参数量: ~200K (官方数据)")
    print(f"FLOPs: 未公开")


if __name__ == '__main__':
    main()