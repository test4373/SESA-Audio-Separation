# coding: utf-8
__author__ = 'PyTorch Backend Implementation'

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
import warnings
import hashlib
import time


class PyTorchBackend:
    """
    Optimized PyTorch backend for model inference.
    Provides various optimization techniques for faster inference.
    """
    
    def __init__(self, device='cuda:0', optimize_mode='default'):
        """
        Initialize PyTorch backend.
        
        Parameters:
        ----------
        device : str
            Device to use for inference (cuda:0, cpu, mps, etc.)
        optimize_mode : str
            Optimization mode: 'default', 'compile', 'jit', or 'channels_last'
        """
        self.device = device
        self.optimize_mode = optimize_mode
        self.model = None
        self.compiled_model = None
        
        # Check device availability
        if device.startswith('cuda') and not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        elif device == 'mps' and not torch.backends.mps.is_available():
            warnings.warn("MPS not available, falling back to CPU")
            self.device = 'cpu'
    
    def optimize_model(
        self,
        model: nn.Module,
        example_input: Optional[torch.Tensor] = None,
        use_amp: bool = True,
        use_channels_last: bool = False
    ) -> nn.Module:
        """
        Optimize PyTorch model for inference.
        
        Parameters:
        ----------
        model : nn.Module
            PyTorch model to optimize
        example_input : Optional[torch.Tensor]
            Example input for optimization (required for some modes)
        use_amp : bool
            Use automatic mixed precision (AMP)
        use_channels_last : bool
            Use channels-last memory format
            
        Returns:
        -------
        nn.Module
            Optimized model
        """
        print(f"Optimizing model with mode: {self.optimize_mode}")
        
        self.model = model.eval().to(self.device)
        self.use_amp = use_amp
        
        # Apply memory format optimization
        if use_channels_last and self.device.startswith('cuda'):
            print("  → Applying channels-last memory format")
            self.model = self.model.to(memory_format=torch.channels_last)
        
        # Apply optimization based on mode
        if self.optimize_mode == 'compile':
            self.compiled_model = self._compile_model(model)
        elif self.optimize_mode == 'jit':
            if example_input is None:
                raise ValueError("example_input required for JIT optimization")
            self.compiled_model = self._jit_trace_model(model, example_input)
        elif self.optimize_mode == 'channels_last':
            print("  → Using channels-last optimization")
            self.model = self.model.to(memory_format=torch.channels_last)
            self.compiled_model = self.model
        else:
            print("  → Using default optimization")
            self.compiled_model = self.model
        
        print("✓ Model optimization complete")
        return self.compiled_model
    
    def _compile_model(self, model: nn.Module) -> nn.Module:
        """
        Compile model using torch.compile (PyTorch 2.0+).
        
        Parameters:
        ----------
        model : nn.Module
            Model to compile
            
        Returns:
        -------
        nn.Module
            Compiled model
        """
        try:
            if hasattr(torch, 'compile'):
                print("  → Compiling model with torch.compile")
                compiled = torch.compile(model, mode='reduce-overhead')
                return compiled
            else:
                print("  ⚠ torch.compile not available (requires PyTorch 2.0+)")
                return model
        except Exception as e:
            print(f"  ⚠ Compilation failed: {e}")
            return model
    
    def _jit_trace_model(self, model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        """
        Trace model using TorchScript JIT.
        
        Parameters:
        ----------
        model : nn.Module
            Model to trace
        example_input : torch.Tensor
            Example input for tracing
            
        Returns:
        -------
        nn.Module
            Traced model
        """
        try:
            print("  → Tracing model with TorchScript JIT")
            with torch.no_grad():
                traced = torch.jit.trace(model, example_input)
            traced = torch.jit.optimize_for_inference(traced)
            return traced
        except Exception as e:
            print(f"  ⚠ JIT tracing failed: {e}")
            return model
    
    def save_optimized_model(self, save_path: str):
        """
        Save optimized model to file.
        
        Parameters:
        ----------
        save_path : str
            Path to save the model
        """
        if self.compiled_model is None:
            raise RuntimeError("No model has been optimized yet")
        
        try:
            # Save based on optimization mode
            if self.optimize_mode == 'jit':
                torch.jit.save(self.compiled_model, save_path)
            else:
                torch.save(self.compiled_model.state_dict(), save_path)
            print(f"✓ Model saved to: {save_path}")
        except Exception as e:
            print(f"✗ Failed to save model: {e}")
    
    def load_optimized_model(self, load_path: str, model_template: nn.Module) -> nn.Module:
        """
        Load optimized model from file.
        
        Parameters:
        ----------
        load_path : str
            Path to the saved model
        model_template : nn.Module
            Model template for loading state dict
            
        Returns:
        -------
        nn.Module
            Loaded model
        """
        try:
            if self.optimize_mode == 'jit':
                self.compiled_model = torch.jit.load(load_path, map_location=self.device)
            else:
                model_template.load_state_dict(torch.load(load_path, map_location=self.device))
                self.compiled_model = model_template.eval()
            
            print(f"✓ Model loaded from: {load_path}")
            return self.compiled_model
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference with optimized model.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns:
        -------
        torch.Tensor
            Model output
        """
        if self.compiled_model is None:
            raise RuntimeError("No model has been optimized yet")
        
        # Apply memory format if needed
        if self.optimize_mode == 'channels_last':
            x = x.to(memory_format=torch.channels_last)
        
        # Run inference with AMP if enabled
        if self.use_amp and self.device.startswith('cuda'):
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    return self.compiled_model(x)
        else:
            with torch.no_grad():
                return self.compiled_model(x)


class PyTorchOptimizer:
    """
    Helper class for various PyTorch optimization techniques.
    """
    
    @staticmethod
    def enable_cudnn_benchmark():
        """Enable cuDNN benchmark mode for faster inference."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            print("✓ cuDNN benchmark enabled")
    
    @staticmethod
    def enable_cudnn_deterministic():
        """Enable cuDNN deterministic mode for reproducible results."""
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print("✓ cuDNN deterministic mode enabled")
    
    @staticmethod
    def enable_tf32():
        """Enable TF32 for Ampere GPUs (RTX 30xx+)."""
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✓ TF32 enabled")
    
    @staticmethod
    def set_num_threads(num_threads: int):
        """Set number of threads for CPU inference."""
        torch.set_num_threads(num_threads)
        print(f"✓ Number of threads set to: {num_threads}")
    
    @staticmethod
    def optimize_for_inference(model: nn.Module) -> nn.Module:
        """
        Apply general optimization for inference.
        
        Parameters:
        ----------
        model : nn.Module
            Model to optimize
            
        Returns:
        -------
        nn.Module
            Optimized model
        """
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        # Fuse batch norm if possible
        try:
            model = torch.quantization.fuse_modules(model, inplace=True)
            print("✓ Batch norm fused")
        except:
            pass
        
        return model


def benchmark_pytorch_optimizations(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cuda:0',
    num_iterations: int = 100,
    warmup_iterations: int = 10
) -> Dict[str, float]:
    """
    Benchmark different PyTorch optimization techniques.
    
    Parameters:
    ----------
    model : nn.Module
        Model to benchmark
    input_shape : Tuple[int, ...]
        Input tensor shape
    device : str
        Device to use
    num_iterations : int
        Number of benchmark iterations
    warmup_iterations : int
        Number of warmup iterations
        
    Returns:
    -------
    Dict[str, float]
        Benchmark results with average inference times
    """
    results = {}
    dummy_input = torch.randn(*input_shape).to(device)
    
    optimization_modes = ['default', 'compile', 'channels_last']
    
    for mode in optimization_modes:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {mode}")
        print('='*60)
        
        try:
            backend = PyTorchBackend(device=device, optimize_mode=mode)
            
            # Optimize model
            if mode == 'compile':
                optimized_model = backend.optimize_model(model, use_amp=True)
            else:
                optimized_model = backend.optimize_model(
                    model, 
                    example_input=dummy_input,
                    use_amp=True,
                    use_channels_last=(mode == 'channels_last')
                )
            
            # Warmup
            for _ in range(warmup_iterations):
                _ = backend(dummy_input)
            
            # Benchmark
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            
            start = time.time()
            for _ in range(num_iterations):
                _ = backend(dummy_input)
            
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            
            elapsed = (time.time() - start) / num_iterations
            results[mode] = elapsed * 1000  # Convert to ms
            
            print(f"  Average time: {results[mode]:.2f} ms")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results[mode] = None
    
    return results


def create_inference_session(
    model: nn.Module,
    device: str = 'cuda:0',
    optimize_mode: str = 'default',
    enable_amp: bool = True,
    enable_tf32: bool = True,
    enable_cudnn_benchmark: bool = True
) -> PyTorchBackend:
    """
    Create an optimized inference session.
    
    Parameters:
    ----------
    model : nn.Module
        Model to use for inference
    device : str
        Device to use
    optimize_mode : str
        Optimization mode
    enable_amp : bool
        Enable automatic mixed precision
    enable_tf32 : bool
        Enable TF32 (for Ampere GPUs)
    enable_cudnn_benchmark : bool
        Enable cuDNN benchmark
        
    Returns:
    -------
    PyTorchBackend
        Configured inference session
    """
    # Apply global optimizations
    optimizer = PyTorchOptimizer()
    
    if enable_cudnn_benchmark:
        optimizer.enable_cudnn_benchmark()
    
    if enable_tf32 and device.startswith('cuda'):
        optimizer.enable_tf32()
    
    # Create backend
    backend = PyTorchBackend(device=device, optimize_mode=optimize_mode)
    backend.optimize_model(model, use_amp=enable_amp)
    
    return backend


def convert_model_to_onnx(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    output_path: str,
    opset_version: int = 14
):
    """
    Convert PyTorch model to ONNX format.
    
    Parameters:
    ----------
    model : nn.Module
        Model to convert
    input_shape : Tuple[int, ...]
        Input tensor shape
    output_path : str
        Path to save ONNX model
    opset_version : int
        ONNX opset version
    """
    try:
        import onnx
        
        model.eval()
        dummy_input = torch.randn(*input_shape)
        
        print(f"Converting model to ONNX (opset {opset_version})...")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"✓ ONNX model saved to: {output_path}")
        
    except ImportError:
        print("✗ ONNX not available. Install with: pip install onnx")
    except Exception as e:
        print(f"✗ ONNX conversion failed: {e}")


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about a PyTorch model.
    
    Parameters:
    ----------
    model : nn.Module
        Model to analyze
        
    Returns:
    -------
    Dict[str, Any]
        Model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': size_mb,
        'device': next(model.parameters()).device,
        'dtype': next(model.parameters()).dtype
    }
