# coding: utf-8
__author__ = 'PyTorch Optimization Benchmark Tool'

import argparse
import time
import torch
import numpy as np
from utils import get_model_from_config
from pytorch_backend import (
    PyTorchBackend, 
    PyTorchOptimizer, 
    benchmark_pytorch_optimizations,
    get_model_info
)
import sys


def load_checkpoint(checkpoint_path: str, model, device: str):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state' in checkpoint:
            state_dict = checkpoint['state']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model = model.eval().to(device)
    
    print("✓ Checkpoint loaded successfully")
    return model


def benchmark_optimization_modes(args):
    """
    Benchmark different PyTorch optimization modes.
    """
    parser = argparse.ArgumentParser(description="Benchmark PyTorch Optimization Modes")
    parser.add_argument("--model_type", type=str, required=True, help="Model type")
    parser.add_argument("--config_path", type=str, required=True, help="Config path")
    parser.add_argument("--start_check_point", type=str, required=True, help="Checkpoint path (.ckpt)")
    parser.add_argument("--device", type=str, default='cuda:0', help="Device")
    parser.add_argument("--num_iterations", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--warmup_iterations", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--chunk_size", type=int, default=None, help="Override chunk size (optional)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    
    # Check device
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        return
    
    print("="*60)
    print("PyTorch Optimization Benchmark Tool")
    print("="*60)
    print(f"Model Type: {args.model_type}")
    print(f"Checkpoint: {args.start_check_point}")
    print(f"Device: {args.device}")
    print(f"Iterations: {args.num_iterations}")
    print("="*60)
    
    # Load model
    print("\n📦 Loading model...")
    model, config = get_model_from_config(args.model_type, args.config_path)
    model = load_checkpoint(args.start_check_point, model, args.device)
    
    # Get model info
    model_info = get_model_info(model)
    print(f"\n📊 Model Information:")
    print(f"  Total Parameters: {model_info['total_parameters']:,}")
    print(f"  Trainable Parameters: {model_info['trainable_parameters']:,}")
    print(f"  Model Size: {model_info['model_size_mb']:.2f} MB")
    print(f"  Device: {model_info['device']}")
    print(f"  Dtype: {model_info['dtype']}")
    
    # Get chunk size
    if args.chunk_size:
        chunk_size = args.chunk_size
    else:
        chunk_size = config.audio.chunk_size
    
    num_channels = 2
    input_shape = (args.batch_size, num_channels, chunk_size)
    
    print(f"\n📊 Test Configuration:")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Channels: {num_channels}")
    print(f"  Chunk Size: {chunk_size}")
    print(f"  Input Shape: {input_shape}")
    
    # Benchmark different optimization modes
    print("\n" + "="*60)
    print("Benchmarking Optimization Modes")
    print("="*60)
    
    results = benchmark_pytorch_optimizations(
        model=model,
        input_shape=input_shape,
        device=args.device,
        num_iterations=args.num_iterations,
        warmup_iterations=args.warmup_iterations
    )
    
    # Display results
    print("\n" + "="*60)
    print("📈 Benchmark Results")
    print("="*60)
    
    baseline = None
    for mode, time_ms in results.items():
        if time_ms is not None:
            if baseline is None:
                baseline = time_ms
            speedup = baseline / time_ms if time_ms > 0 else 0
            improvement = ((baseline - time_ms) / baseline) * 100 if baseline > 0 else 0
            
            print(f"\n{mode.upper()}:")
            print(f"  Average Time: {time_ms:.2f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Improvement: {improvement:.1f}%")
    
    print("\n" + "="*60)
    
    # Recommendations
    print("\n💡 Recommendations:")
    
    if results.get('compile') and results['compile'] < results['default']:
        print("  ✓ Use 'compile' mode for best performance (PyTorch 2.0+)")
    elif results.get('channels_last') and results['channels_last'] < results['default']:
        print("  ✓ Use 'channels_last' mode for better performance")
    else:
        print("  ✓ Default mode is optimal for your configuration")
    
    if args.device.startswith('cuda'):
        print("  ✓ Enable TF32 for Ampere GPUs (RTX 30xx+)")
        print("  ✓ Enable cuDNN benchmark for consistent input sizes")
    
    print("\n✅ Benchmark completed!")


def test_optimization_modes(args):
    """
    Test different optimization modes with verification.
    """
    parser = argparse.ArgumentParser(description="Test PyTorch Optimization Modes")
    parser.add_argument("--model_type", type=str, required=True, help="Model type")
    parser.add_argument("--config_path", type=str, required=True, help="Config path")
    parser.add_argument("--start_check_point", type=str, required=True, help="Checkpoint path (.ckpt)")
    parser.add_argument("--device", type=str, default='cuda:0', help="Device")
    
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    
    print("="*60)
    print("PyTorch Optimization Mode Test")
    print("="*60)
    
    # Load model
    print("\n📦 Loading model...")
    model, config = get_model_from_config(args.model_type, args.config_path)
    model = load_checkpoint(args.start_check_point, model, args.device)
    
    chunk_size = config.audio.chunk_size
    input_shape = (1, 2, chunk_size)
    dummy_input = torch.randn(*input_shape).to(args.device)
    
    # Test each optimization mode
    modes = ['default', 'compile', 'channels_last']
    outputs = {}
    
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Testing: {mode}")
        print('='*60)
        
        try:
            backend = PyTorchBackend(device=args.device, optimize_mode=mode)
            
            if mode == 'jit':
                backend.optimize_model(model, example_input=dummy_input, use_amp=True)
            else:
                backend.optimize_model(
                    model, 
                    use_amp=True,
                    use_channels_last=(mode == 'channels_last')
                )
            
            # Run inference
            with torch.no_grad():
                output = backend(dummy_input)
            
            outputs[mode] = output
            print(f"✓ {mode} successful")
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: [{output.min().item():.6f}, {output.max().item():.6f}]")
            
        except Exception as e:
            print(f"✗ {mode} failed: {e}")
            outputs[mode] = None
    
    # Verify outputs match
    print("\n" + "="*60)
    print("🔍 Output Verification")
    print("="*60)
    
    baseline_key = 'default'
    if baseline_key in outputs and outputs[baseline_key] is not None:
        baseline_output = outputs[baseline_key]
        
        for mode, output in outputs.items():
            if mode != baseline_key and output is not None:
                diff = torch.abs(baseline_output - output)
                max_diff = torch.max(diff).item()
                mean_diff = torch.mean(diff).item()
                
                print(f"\n{mode} vs {baseline_key}:")
                print(f"  Max difference: {max_diff:.6f}")
                print(f"  Mean difference: {mean_diff:.6f}")
                
                if max_diff < 1e-3:
                    print(f"  ✓ Outputs match within tolerance")
                else:
                    print(f"  ⚠ Warning: Large difference detected!")
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        sys.argv.pop(1)
        test_optimization_modes(None)
    else:
        benchmark_optimization_modes(None)
