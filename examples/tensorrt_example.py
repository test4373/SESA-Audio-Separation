#!/usr/bin/env python
# coding: utf-8
"""
TensorRT Backend Kullanım Örnekleri
Bu dosya TensorRT backend'inin farklı kullanım senaryolarını gösterir.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import numpy as np
from tensorrt_backend import (
    TensorRTBackend, 
    is_tensorrt_available, 
    convert_model_to_tensorrt,
    load_model_from_checkpoint
)
from utils import get_model_from_config


def example_1_basic_usage():
    """
    Örnek 1: Temel TensorRT Backend Kullanımı
    """
    print("="*60)
    print("Örnek 1: Temel TensorRT Backend Kullanımı")
    print("="*60)
    
    # TensorRT kontrolü
    if not is_tensorrt_available():
        print("❌ TensorRT mevcut değil!")
        return
    
    # Model yükle
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config.yaml'
    checkpoint_path = 'ckpts/model.ckpt'
    
    print("📦 Model yükleniyor...")
    model, config = get_model_from_config(model_type, config_path)
    model = load_model_from_checkpoint(checkpoint_path, model, 'cuda:0')
    
    # TensorRT backend oluştur
    print("🔧 TensorRT backend oluşturuluyor...")
    trt_backend = TensorRTBackend(device='cuda:0')
    
    # Model compile et
    chunk_size = config.audio.chunk_size
    input_shape = (1, 2, chunk_size)  # (batch, channels, samples)
    
    print(f"⚙️  Model compile ediliyor (input shape: {input_shape})...")
    trt_backend.compile_model(
        model=model,
        input_shape=input_shape,
        precision='fp16',
        use_cache=True
    )
    
    # Test inference
    print("🧪 Test inference...")
    dummy_input = torch.randn(*input_shape).cuda()
    
    with torch.no_grad():
        output = trt_backend(dummy_input)
    
    print(f"✓ Çıktı şekli: {output.shape}")
    print("✅ Örnek 1 tamamlandı!\n")


def example_2_performance_comparison():
    """
    Örnek 2: PyTorch vs TensorRT Performans Karşılaştırması
    """
    print("="*60)
    print("Örnek 2: PyTorch vs TensorRT Performans Karşılaştırması")
    print("="*60)
    
    if not is_tensorrt_available():
        print("❌ TensorRT mevcut değil!")
        return
    
    # Model yükle
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config.yaml'
    checkpoint_path = 'ckpts/model.ckpt'
    
    print("📦 Model yükleniyor...")
    model, config = get_model_from_config(model_type, config_path)
    model = load_model_from_checkpoint(checkpoint_path, model, 'cuda:0')
    model.eval()
    
    # Test parametreleri
    chunk_size = config.audio.chunk_size
    input_shape = (1, 2, chunk_size)
    dummy_input = torch.randn(*input_shape).cuda()
    num_iterations = 50
    warmup = 10
    
    # PyTorch benchmark
    print("\n🔥 PyTorch benchmark...")
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start) / num_iterations
    
    print(f"  PyTorch ortalama: {pytorch_time*1000:.2f} ms")
    
    # TensorRT benchmark
    print("\n🚀 TensorRT benchmark...")
    trt_backend = TensorRTBackend(device='cuda:0')
    trt_backend.compile_model(
        model=model,
        input_shape=input_shape,
        precision='fp16',
        use_cache=False
    )
    
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = trt_backend(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iterations):
            _ = trt_backend(dummy_input)
        torch.cuda.synchronize()
        tensorrt_time = (time.time() - start) / num_iterations
    
    print(f"  TensorRT ortalama: {tensorrt_time*1000:.2f} ms")
    
    # Karşılaştırma
    speedup = pytorch_time / tensorrt_time
    print(f"\n📈 Sonuç:")
    print(f"  Hızlanma: {speedup:.2f}x")
    print(f"  İyileştirme: {((pytorch_time-tensorrt_time)/pytorch_time)*100:.1f}%")
    print("✅ Örnek 2 tamamlandı!\n")


def example_3_save_and_load_engine():
    """
    Örnek 3: TensorRT Engine Kaydetme ve Yükleme
    """
    print("="*60)
    print("Örnek 3: TensorRT Engine Kaydetme ve Yükleme")
    print("="*60)
    
    if not is_tensorrt_available():
        print("❌ TensorRT mevcut değil!")
        return
    
    # Model yükle
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config.yaml'
    checkpoint_path = 'ckpts/model.ckpt'
    
    print("📦 Model yükleniyor...")
    model, config = get_model_from_config(model_type, config_path)
    model = load_model_from_checkpoint(checkpoint_path, model, 'cuda:0')
    
    # TensorRT compile
    print("🔧 Model compile ediliyor...")
    trt_backend = TensorRTBackend(device='cuda:0')
    chunk_size = config.audio.chunk_size
    input_shape = (1, 2, chunk_size)
    
    trt_backend.compile_model(
        model=model,
        input_shape=input_shape,
        precision='fp16',
        use_cache=False
    )
    
    # Engine'i kaydet
    save_path = 'trt_cache/example_model.trt'
    print(f"💾 Engine kaydediliyor: {save_path}")
    trt_backend.save_engine(save_path)
    
    # Yeni backend oluştur ve engine'i yükle
    print("📥 Engine yükleniyor...")
    new_backend = TensorRTBackend(device='cuda:0')
    new_backend.load_engine(save_path)
    
    # Test
    print("🧪 Test inference...")
    dummy_input = torch.randn(*input_shape).cuda()
    with torch.no_grad():
        output = new_backend(dummy_input)
    
    print(f"✓ Çıktı şekli: {output.shape}")
    print("✅ Örnek 3 tamamlandı!\n")


def example_4_batch_processing():
    """
    Örnek 4: Batch Processing ile TensorRT
    """
    print("="*60)
    print("Örnek 4: Batch Processing ile TensorRT")
    print("="*60)
    
    if not is_tensorrt_available():
        print("❌ TensorRT mevcut değil!")
        return
    
    # Model yükle
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config.yaml'
    checkpoint_path = 'ckpts/model.ckpt'
    
    print("📦 Model yükleniyor...")
    model, config = get_model_from_config(model_type, config_path)
    model = load_model_from_checkpoint(checkpoint_path, model, 'cuda:0')
    
    # Farklı batch size'lar için test
    chunk_size = config.audio.chunk_size
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        print(f"\n📊 Batch Size: {batch_size}")
        
        # TensorRT compile
        input_shape = (batch_size, 2, chunk_size)
        trt_backend = TensorRTBackend(device='cuda:0')
        trt_backend.compile_model(
            model=model,
            input_shape=input_shape,
            precision='fp16',
            use_cache=False
        )
        
        # Test
        dummy_input = torch.randn(*input_shape).cuda()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = trt_backend(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(50):
                _ = trt_backend(dummy_input)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 50
        
        print(f"  Ortalama süre: {elapsed*1000:.2f} ms")
        print(f"  Sample başına: {(elapsed*1000)/batch_size:.2f} ms")
    
    print("\n✅ Örnek 4 tamamlandı!\n")


def example_5_precision_comparison():
    """
    Örnek 5: FP32 vs FP16 Precision Karşılaştırması
    """
    print("="*60)
    print("Örnek 5: FP32 vs FP16 Precision Karşılaştırması")
    print("="*60)
    
    if not is_tensorrt_available():
        print("❌ TensorRT mevcut değil!")
        return
    
    # Model yükle
    model_type = 'mel_band_roformer'
    config_path = 'ckpts/config.yaml'
    checkpoint_path = 'ckpts/model.ckpt'
    
    print("📦 Model yükleniyor...")
    model, config = get_model_from_config(model_type, config_path)
    model = load_model_from_checkpoint(checkpoint_path, model, 'cuda:0')
    
    chunk_size = config.audio.chunk_size
    input_shape = (1, 2, chunk_size)
    dummy_input = torch.randn(*input_shape).cuda()
    
    results = {}
    
    for precision in ['fp32', 'fp16']:
        print(f"\n🔧 {precision.upper()} precision ile compile ediliyor...")
        
        trt_backend = TensorRTBackend(device='cuda:0')
        trt_backend.compile_model(
            model=model,
            input_shape=input_shape,
            precision=precision,
            use_cache=False
        )
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = trt_backend(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(50):
                output = trt_backend(dummy_input)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 50
        
        results[precision] = {
            'time': elapsed * 1000,
            'output': output
        }
        
        print(f"  Ortalama süre: {elapsed*1000:.2f} ms")
    
    # Doğruluk karşılaştırması
    print("\n🔍 Doğruluk karşılaştırması:")
    fp32_out = results['fp32']['output']
    fp16_out = results['fp16']['output']
    
    diff = torch.abs(fp32_out - fp16_out)
    print(f"  Max fark: {torch.max(diff).item():.6f}")
    print(f"  Mean fark: {torch.mean(diff).item():.6f}")
    
    # Hız karşılaştırması
    speedup = results['fp32']['time'] / results['fp16']['time']
    print(f"\n📈 FP16 hızlanma: {speedup:.2f}x")
    
    print("\n✅ Örnek 5 tamamlandı!\n")


def main():
    """
    Tüm örnekleri çalıştır
    """
    print("\n" + "="*60)
    print("TensorRT Backend Kullanım Örnekleri")
    print("="*60 + "\n")
    
    if not is_tensorrt_available():
        print("❌ TensorRT mevcut değil!")
        print("Kurulum için: pip install torch2trt nvidia-tensorrt")
        return
    
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
    print()
    
    try:
        # Örnek 1: Temel kullanım
        example_1_basic_usage()
        
        # Örnek 2: Performans karşılaştırması
        example_2_performance_comparison()
        
        # Örnek 3: Save/Load
        example_3_save_and_load_engine()
        
        # Örnek 4: Batch processing
        example_4_batch_processing()
        
        # Örnek 5: Precision karşılaştırması
        example_5_precision_comparison()
        
        print("\n" + "="*60)
        print("🎉 Tüm örnekler başarıyla tamamlandı!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
