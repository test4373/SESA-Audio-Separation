# coding: utf-8
__author__ = 'TensorRT Backend Implementation'

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
import warnings
import hashlib

try:
    import tensorrt as trt
    from torch2trt import torch2trt, TRTModule
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    warnings.warn("TensorRT not available. Install with: pip install torch2trt")


class TensorRTBackend:
    """
    TensorRT backend for optimized model inference.
    Converts PyTorch models to TensorRT for faster inference.
    Works with .ckpt checkpoint files.
    """
    
    def __init__(self, device='cuda:0', cache_dir='trt_cache'):
        """
        Initialize TensorRT backend.
        
        Parameters:
        ----------
        device : str
            CUDA device to use for inference
        cache_dir : str
            Directory to cache TensorRT engines
        """
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available. Please install torch2trt")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for TensorRT backend")
        
        self.device = device
        self.trt_model = None
        self.original_model = None
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def compile_model(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        precision: str = 'fp16',
        max_workspace_size: int = 1 << 30,  # 1GB
        max_batch_size: int = 1,
        use_cache: bool = True,
        cache_key: Optional[str] = None
    ) -> nn.Module:
        """
        Compile PyTorch model to TensorRT.
        
        Parameters:
        ----------
        model : nn.Module
            PyTorch model to compile
        input_shape : Tuple[int, ...]
            Input tensor shape (batch, channels, length)
        precision : str
            Precision mode: 'fp32' or 'fp16'
        max_workspace_size : int
            Maximum workspace size in bytes
        max_batch_size : int
            Maximum batch size
        use_cache : bool
            Use cached TensorRT engine if available
        cache_key : Optional[str]
            Custom cache key, auto-generated if None
            
        Returns:
        -------
        nn.Module
            TensorRT compiled model
        """
        print(f"Compiling model to TensorRT with {precision} precision...")
        
        self.original_model = model
        model = model.eval().to(self.device)
        
        # Generate cache key
        if cache_key is None:
            cache_key = self._generate_cache_key(model, input_shape, precision)
        
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.trt")
        
        # Try loading from cache
        if use_cache and os.path.exists(cache_path):
            try:
                print(f"Loading TensorRT engine from cache: {cache_path}")
                self.trt_model = TRTModule()
                self.trt_model.load_state_dict(torch.load(cache_path))
                print("✓ TensorRT engine loaded from cache")
                return self.trt_model
            except Exception as e:
                print(f"Failed to load cached engine: {e}")
                print("Recompiling...")
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape).to(self.device)
        
        # Set precision
        fp16_mode = (precision == 'fp16')
        
        try:
            with torch.no_grad():
                # Compile to TensorRT using torch2trt
                self.trt_model = torch2trt(
                    model,
                    [dummy_input],
                    fp16_mode=fp16_mode,
                    max_workspace_size=max_workspace_size,
                    max_batch_size=max_batch_size,
                    use_onnx=False
                )
            
            # Save to cache
            if use_cache:
                try:
                    torch.save(self.trt_model.state_dict(), cache_path)
                    print(f"✓ TensorRT engine cached to: {cache_path}")
                except Exception as e:
                    print(f"Warning: Failed to cache engine: {e}")
            
            print("✓ Model successfully compiled to TensorRT")
            return self.trt_model
            
        except Exception as e:
            print(f"✗ TensorRT compilation failed: {e}")
            print("Falling back to PyTorch model")
            self.trt_model = model
            return model
    
    def _generate_cache_key(self, model: nn.Module, input_shape: Tuple[int, ...], precision: str) -> str:
        """
        Generate a unique cache key for the model.
        
        Parameters:
        ----------
        model : nn.Module
            Model to generate key for
        input_shape : Tuple[int, ...]
            Input shape
        precision : str
            Precision mode
            
        Returns:
        -------
        str
            Cache key
        """
        # Create a hash based on model architecture and parameters
        model_str = str(model)
        input_str = str(input_shape)
        key_str = f"{model_str}_{input_str}_{precision}"
        
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def save_engine(self, save_path: str):
        """
        Save TensorRT engine to file.
        
        Parameters:
        ----------
        save_path : str
            Path to save the TensorRT engine
        """
        if self.trt_model is None:
            raise RuntimeError("No model has been compiled yet")
        
        try:
            if isinstance(self.trt_model, TRTModule):
                torch.save(self.trt_model.state_dict(), save_path)
            else:
                torch.save(self.trt_model.state_dict(), save_path)
            print(f"✓ TensorRT engine saved to: {save_path}")
        except Exception as e:
            print(f"✗ Failed to save TensorRT engine: {e}")
    
    def load_engine(self, load_path: str) -> nn.Module:
        """
        Load TensorRT engine from file.
        
        Parameters:
        ----------
        load_path : str
            Path to the saved TensorRT engine
            
        Returns:
        -------
        nn.Module
            Loaded TensorRT model
        """
        try:
            self.trt_model = TRTModule()
            self.trt_model.load_state_dict(torch.load(load_path))
            print(f"✓ TensorRT engine loaded from: {load_path}")
            return self.trt_model
        except Exception as e:
            print(f"✗ Failed to load TensorRT engine: {e}")
            raise
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference with TensorRT model.
        
        Parameters:
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns:
        -------
        torch.Tensor
            Model output
        """
        if self.trt_model is None:
            raise RuntimeError("No model has been compiled yet")
        
        with torch.no_grad():
            return self.trt_model(x)


def convert_model_to_tensorrt(
    model: nn.Module,
    example_input_shape: Tuple[int, ...],
    save_path: Optional[str] = None,
    precision: str = 'fp16',
    device: str = 'cuda:0',
    use_cache: bool = True
) -> TensorRTBackend:
    """
    Helper function to convert a PyTorch model to TensorRT.
    
    Parameters:
    ----------
    model : nn.Module
        PyTorch model to convert
    example_input_shape : Tuple[int, ...]
        Shape of example input tensor
    save_path : Optional[str]
        Path to save the TensorRT engine (optional)
    precision : str
        Precision mode: 'fp32' or 'fp16'
    device : str
        CUDA device to use
    use_cache : bool
        Use cached TensorRT engine if available
        
    Returns:
    -------
    TensorRTBackend
        TensorRT backend with compiled model
    """
    backend = TensorRTBackend(device=device)
    backend.compile_model(
        model=model,
        input_shape=example_input_shape,
        precision=precision,
        use_cache=use_cache
    )
    
    if save_path:
        backend.save_engine(save_path)
    
    return backend


def load_model_from_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    device: str = 'cuda:0'
) -> nn.Module:
    """
    Load model weights from .ckpt checkpoint file.
    
    Parameters:
    ----------
    checkpoint_path : str
        Path to .ckpt checkpoint file
    model : nn.Module
        Model architecture to load weights into
    device : str
        Device to load model on
        
    Returns:
    -------
    nn.Module
        Model with loaded weights
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
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
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model = model.eval().to(device)
    
    print("✓ Checkpoint loaded successfully")
    return model


def is_tensorrt_available() -> bool:
    """
    Check if TensorRT is available.
    
    Returns:
    -------
    bool
        True if TensorRT is available, False otherwise
    """
    return TENSORRT_AVAILABLE and torch.cuda.is_available()


def benchmark_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    use_tensorrt: bool = True,
    device: str = 'cuda:0'
) -> Dict[str, float]:
    """
    Benchmark model performance with and without TensorRT.
    
    Parameters:
    ----------
    model : nn.Module
        Model to benchmark
    input_shape : Tuple[int, ...]
        Input tensor shape
    num_iterations : int
        Number of benchmark iterations
    warmup_iterations : int
        Number of warmup iterations
    use_tensorrt : bool
        Compare with TensorRT
    device : str
        CUDA device to use
        
    Returns:
    -------
    Dict[str, float]
        Benchmark results with average inference times
    """
    import time
    
    results = {}
    
    # Benchmark PyTorch
    print("Benchmarking PyTorch model...")
    model = model.eval().to(device)
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_iterations
    results['pytorch_ms'] = pytorch_time * 1000
    
    # Benchmark TensorRT
    if use_tensorrt and is_tensorrt_available():
        print("Benchmarking TensorRT model...")
        backend = TensorRTBackend(device=device)
        trt_model = backend.compile_model(
            model=model,
            input_shape=input_shape,
            precision='fp16'
        )
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = backend(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = backend(dummy_input)
        torch.cuda.synchronize()
        trt_time = (time.time() - start) / num_iterations
        results['tensorrt_ms'] = trt_time * 1000
        results['speedup'] = pytorch_time / trt_time
    
    return results
