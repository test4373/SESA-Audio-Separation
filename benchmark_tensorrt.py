# coding: utf-8
__author__ = 'TensorRT Benchmark Tool'

import argparse
import time
import torch
import numpy as np
from utils import get_model_from_config
from tensorrt_backend import TensorRTBackend, is_tensorrt_available, load_model_from_checkpoint
import sys

def benchmark_inference(args):
    """
    PyTorch ve TensorRT modellerinin performansını karşılaştırır.
    """
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs TensorRT")
    parser.add_argument("--model_type", type=str, required=True, help="Model type")
    parser.add_argument("--config_path", type=str, required=True, help="Config path")
    parser.add_argument("--start_check_point", type=str, required=True, help="Checkpoint path (.ckpt)")
    parser.add_argument("--device", type=str, default='cuda:0', help="CUDA device")
    parser.add_argument("--precision", type=str, choices=['fp32', 'fp16'], default='fp16', help="TensorRT precision")
    parser.add_argument("--num_iterations", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--warmup_iterations", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--chunk_size", type=int, default=None, help="Override chunk size (optional)")
    
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA is required for benchmarking!")
        return
    
    # Check TensorRT
    if not is_tensorrt_available():
        print("❌ TensorRT is not available!")
        print("Please install with: pip install torch2trt")
        return
    
    print("="*60)
    print("TensorRT Benchmark Tool")
    print("="*60)
    print(f"Model Type: {args.model_type}")
    print(f"Checkpoint: {args.start_check_point}")
    print(f"Device: {args.device}")
    print(f"Precision: {args.precision}")
    print(f"Iterations: {args.num_iterations}")
    print("="*60)
    
    # Load model
    print("\n📦 Loading model...")
    model, config = get_model_from_config(args.model_type, args.config_path)
    model = load_model_from_checkpoint(args.start_check_point, model, args.device)
    model.eval()
    
    # Get chunk size
    if args.chunk_size:
        chunk_size = args.chunk_size
    else:
        chunk_size = config.audio.chunk_size
    
    batch_size = config.inference.batch_size
    num_channels = 2
    
    print(f"\n📊 Test Configuration:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Channels: {num_channels}")
    print(f"  Chunk Size: {chunk_size}")
    print(f"  Input Shape: ({batch_size}, {num_channels}, {chunk_size})")
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, num_channels, chunk_size).to(args.device)
    
    # Benchmark PyTorch
    print("\n" + "="*60)
    print("🔥 Benchmarking PyTorch Model")
    print("="*60)
    
    # Warmup
    print(f"Warming up ({args.warmup_iterations} iterations)...")
    with torch.no_grad():
        for _ in range(args.warmup_iterations):
            _ = model(dummy_input)
    
    # Benchmark
    print(f"Running benchmark ({args.num_iterations} iterations)...")
    torch.cuda.synchronize()
    pytorch_times = []
    
    with torch.no_grad():
        for i in range(args.num_iterations):
            start = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            pytorch_times.append(elapsed)
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{args.num_iterations}")
    
    pytorch_mean = np.mean(pytorch_times) * 1000
    pytorch_std = np.std(pytorch_times) * 1000
    pytorch_min = np.min(pytorch_times) * 1000
    pytorch_max = np.max(pytorch_times) * 1000
    
    print(f"\n✓ PyTorch Results:")
    print(f"  Mean: {pytorch_mean:.2f} ms")
    print(f"  Std:  {pytorch_std:.2f} ms")
    print(f"  Min:  {pytorch_min:.2f} ms")
    print(f"  Max:  {pytorch_max:.2f} ms")
    
    # Benchmark TensorRT
    print("\n" + "="*60)
    print("🚀 Benchmarking TensorRT Model")
    print("="*60)
    
    # Compile to TensorRT
    print("Compiling to TensorRT...")
    trt_backend = TensorRTBackend(device=args.device)
    
    try:
        trt_backend.compile_model(
            model=model,
            input_shape=(batch_size, num_channels, chunk_size),
            precision=args.precision,
            use_cache=False
        )
    except Exception as e:
        print(f"❌ TensorRT compilation failed: {e}")
        return
    
    # Warmup
    print(f"Warming up ({args.warmup_iterations} iterations)...")
    with torch.no_grad():
        for _ in range(args.warmup_iterations):
            _ = trt_backend(dummy_input)
    
    # Benchmark
    print(f"Running benchmark ({args.num_iterations} iterations)...")
    torch.cuda.synchronize()
    tensorrt_times = []
    
    with torch.no_grad():
        for i in range(args.num_iterations):
            start = time.time()
            _ = trt_backend(dummy_input)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            tensorrt_times.append(elapsed)
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{args.num_iterations}")
    
    tensorrt_mean = np.mean(tensorrt_times) * 1000
    tensorrt_std = np.std(tensorrt_times) * 1000
    tensorrt_min = np.min(tensorrt_times) * 1000
    tensorrt_max = np.max(tensorrt_times) * 1000
    
    print(f"\n✓ TensorRT Results:")
    print(f"  Mean: {tensorrt_mean:.2f} ms")
    print(f"  Std:  {tensorrt_std:.2f} ms")
    print(f"  Min:  {tensorrt_min:.2f} ms")
    print(f"  Max:  {tensorrt_max:.2f} ms")
    
    # Comparison
    speedup = pytorch_mean / tensorrt_mean
    improvement = ((pytorch_mean - tensorrt_mean) / pytorch_mean) * 100
    
    print("\n" + "="*60)
    print("📈 Comparison")
    print("="*60)
    print(f"  PyTorch:   {pytorch_mean:.2f} ms")
    print(f"  TensorRT:  {tensorrt_mean:.2f} ms")
    print(f"  Speedup:   {speedup:.2f}x")
    print(f"  Improvement: {improvement:.1f}%")
    print("="*60)
    
    # Verify outputs match
    print("\n🔍 Verifying output correctness...")
    with torch.no_grad():
        pytorch_output = model(dummy_input)
        tensorrt_output = trt_backend(dummy_input)
        
        # Calculate difference
        diff = torch.abs(pytorch_output - tensorrt_output)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"  Max difference:  {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        
        if max_diff < 1e-2:
            print("  ✓ Outputs match within tolerance")
        else:
            print("  ⚠ Warning: Large difference detected!")
    
    print("\n✅ Benchmark completed!")


if __name__ == "__main__":
    benchmark_inference(None)
