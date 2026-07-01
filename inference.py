# inference.py - 完整版：推理 + FLOPs + 内存
import torch
import numpy as np
from pathlib import Path
import argparse
import time
import yaml
import psutil
import os
from tqdm import tqdm

from models.unet import TinyDenoiser
from dataset.utils import read_exr_rgb, write_exr_rgb, to_log_domain, from_log_domain

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False


def get_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def get_flops(model, input_tensor):
    if HAS_FVCORE:
        flops = FlopCountAnalysis(model, input_tensor)
        return flops.total() / 1e6
    return -1


class Denoiser:
    def __init__(self, model_path, config_path='config.yaml', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        model_config = self.config['model']
        print(f"Model config: features={model_config.get('features', [32, 48, 64])}")

        self.model = TinyDenoiser(
            in_ch=3,
            out_ch=3,
            features=model_config.get('features', [32, 48, 64]),
            use_scene_bias=model_config.get('use_scene_bias', False),
            scene_bias_config=model_config.get('scene_bias_config', None)
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', 'unknown')
            val_loss = checkpoint.get('best_val_loss', 'unknown')
            print(f"Loaded checkpoint - Epoch: {epoch}, Val Loss: {val_loss}")
        elif 'ema_shadow' in checkpoint:
            state_dict = checkpoint['ema_shadow']
            print("Loaded EMA shadow weights")
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        self.epsilon = self.config['data']['exr_epsilon']
        params = sum(p.numel() for p in self.model.parameters())
        model_size_mb = get_model_size(self.model)

        print(f"Model parameters: {params:,} ({params / 1000:.1f}K)")
        print(f"Model size: {model_size_mb:.2f} MB")
        print(f"Epsilon: {self.epsilon}")
        print(f"Using device: {self.device}")

        print("Warming up...")
        dummy = torch.randn(1, 3, 512, 512).to(self.device)
        for _ in range(3):
            _ = self.model(dummy)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        self.flops = get_flops(self.model, dummy)
        if self.flops > 0:
            print(f"FLOPs: {self.flops:.2f} MFLOPs (512x512)")
        else:
            print("FLOPs: N/A (install fvcore for FLOPs)")

        mem_before = get_memory_usage()
        _ = self.model(dummy)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        mem_after = get_memory_usage()
        self.mem_inference = mem_after - mem_before
        print(f"Inference memory: {self.mem_inference:.2f} MB")

        if self.device.type == 'cuda':
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    @torch.no_grad()
    def denoise(self, input_exr_path, output_exr_path=None):
        img = read_exr_rgb(input_exr_path)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)

        img_log = to_log_domain(img_tensor, self.epsilon)
        img_log = img_log.to(self.device)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()

        pred_log = self.model(img_log)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        inference_time = (time.time() - start_time) * 1000

        pred_linear = from_log_domain(pred_log, self.epsilon)
        pred_linear = pred_linear.squeeze(0).cpu().numpy()

        if output_exr_path:
            write_exr_rgb(output_exr_path, pred_linear)

        return pred_linear, inference_time

    def batch_denoise(self, input_dir, output_dir, pattern="*.exr"):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exr_files = list(input_dir.rglob(pattern))
        exr_files = [f for f in exr_files if 'ground_truth' not in f.name]

        if not exr_files:
            print(f"No EXR files found in {input_dir}")
            return

        total_time = 0
        print(f"Found {len(exr_files)} EXR files")

        for exr_file in tqdm(exr_files, desc="Denoising"):
            rel_path = exr_file.relative_to(input_dir)
            output_path = output_dir / rel_path.parent / f"denoised_{rel_path.name}"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            _, infer_time = self.denoise(exr_file, output_path)
            total_time += infer_time

        avg_time = total_time / len(exr_files)
        print(f"\nBatch denoising completed!")
        print(f"   Average inference time: {avg_time:.2f}ms per image")
        print(f"   Total: {total_time:.1f}ms for {len(exr_files)} images")
        print(f"   Results saved to: {output_dir}")

    def print_benchmark(self):
        print(f"\n{'='*50}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*50}")
        print(f"  Parameters:     {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Model size:     {get_model_size(self.model):.2f} MB")
        if self.flops > 0:
            print(f"  FLOPs:          {self.flops:.2f} MFLOPs (512x512)")
        print(f"  Inference RAM:  {self.mem_inference:.2f} MB")

    def denoise_and_compare(self, input_exr_path, output_exr_path=None):
        pred, inference_time = self.denoise(input_exr_path, output_exr_path)
        original = read_exr_rgb(input_exr_path)

        print(f"\nDenoising Statistics:")
        print(f"   Input  range: [{original.min():.4f}, {original.max():.4f}]")
        print(f"   Output range: [{pred.min():.4f}, {pred.max():.4f}]")
        print(f"   Inference time: {inference_time:.2f}ms")

        return pred, inference_time


def main():
    parser = argparse.ArgumentParser(description='MC Denoiser Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--input', type=str, required=True, help='Input EXR file or directory')
    parser.add_argument('--output', type=str, default=None, help='Output EXR file or directory')
    parser.add_argument('--batch', action='store_true', help='Batch processing mode')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--benchmark', action='store_true', help='Print benchmark summary')
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input {args.input} does not exist")
        return

    denoiser = Denoiser(
        model_path=args.model,
        config_path=args.config,
        device=args.device
    )

    if args.benchmark:
        denoiser.print_benchmark()

    if args.batch:
        output_dir = args.output if args.output else "denoised_results"
        denoiser.batch_denoise(args.input, output_dir)
    else:
        output_path = args.output if args.output else "denoised.exr"
        denoiser.denoise_and_compare(args.input, output_path)
        print(f"\nDenoised image saved to {output_path}")


if __name__ == '__main__':
    main()