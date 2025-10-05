# TensorRT Kurulum Rehberi

Bu rehber, projedeki TensorRT backend'ini kurmanız için adım adım talimatlar içerir.

## 🎯 Gereksinimler

- NVIDIA GPU (Compute Capability 6.0+)
- Windows 10/11 veya Linux
- Python 3.8+
- CUDA 11.x veya 12.x
- PyTorch with CUDA support

## 📋 Adım Adım Kurulum

### 1. CUDA Toolkit Kontrolü

CUDA'nın yüklü olduğunu kontrol edin:

```bash
nvcc --version
```

Eğer CUDA yüklü değilse:
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) indirin ve kurun

### 2. PyTorch CUDA Kontrolü

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Çıktı `True` olmalıdır. Eğer `False` ise, PyTorch'u CUDA desteği ile yeniden kurun:

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. TensorRT Kurulumu

#### Yöntem 1: pip ile (Önerilen)

```bash
pip install nvidia-tensorrt
pip install torch2trt
```

#### Yöntem 2: Manuel Kurulum

1. [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) sayfasından indirin
2. İndirilen dosyayı extract edin
3. Python wheel'i kurun:

```bash
cd TensorRT-x.x.x.x/python
pip install tensorrt-*-cp3x-none-win_amd64.whl
```

### 4. torch2trt Kurulumu

```bash
# GitHub'dan klon
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install
```

### 5. Kurulum Testi

```bash
python -c "from tensorrt_backend import is_tensorrt_available; print('TensorRT Available:', is_tensorrt_available())"
```

Çıktı `True` olmalıdır.

## 🔧 Kurulum Sorunları

### ImportError: No module named 'tensorrt'

**Çözüm:**
```bash
pip install nvidia-tensorrt --force-reinstall
```

### torch2trt ImportError

**Çözüm:**
```bash
# Önce torch2trt'yi kaldırın
pip uninstall torch2trt

# GitHub'dan yeniden kurun
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install
```

### CUDA Version Mismatch

TensorRT ve PyTorch CUDA versiyonları uyumlu olmalıdır:

```bash
# PyTorch CUDA versiyonunu kontrol edin
python -c "import torch; print(torch.version.cuda)"

# Eğer uyumsuzluk varsa, uyumlu PyTorch kurun
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Windows'ta DLL Eksik Hatası

`cudnn64_8.dll` veya `nvinfer.dll` bulunamıyorsa:

1. CUDA Toolkit'in `bin` klasörünü PATH'e ekleyin:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
```

2. TensorRT'nin `lib` klasörünü PATH'e ekleyin:
```
C:\TensorRT-x.x.x.x\lib
```

### Linux'ta Permission Error

```bash
sudo pip install nvidia-tensorrt torch2trt
```

## ✅ Kurulum Doğrulama

### Test Script Çalıştırma

Proje klasöründe:

```bash
python benchmark_tensorrt.py \
    --model_type mel_band_roformer \
    --config_path ckpts/config.yaml \
    --start_check_point ckpts/model.ckpt \
    --num_iterations 10
```

Bu komut:
- ✓ Model yükleme
- ✓ TensorRT compilation
- ✓ Inference
- ✓ Benchmark

adımlarını test eder.

## 🚀 Hızlı Başlangıç

TensorRT kurulumundan sonra:

```bash
# TensorRT ile inference
python inference_tensorrt.py \
    --model_type mel_band_roformer \
    --config_path ckpts/config.yaml \
    --start_check_point ckpts/model.ckpt \
    --input_folder input/ \
    --store_dir output/ \
    --tensorrt_precision fp16 \
    --use_tensorrt_cache
```

## 📦 Tüm Gereksinimler

```bash
# Tüm bağımlılıkları kur
pip install -r requirements.txt

# TensorRT paketlerini manuel kur
pip install nvidia-tensorrt
pip install torch2trt
```

## 🔍 Sistem Bilgilerini Kontrol Etme

```python
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

try:
    import tensorrt as trt
    print(f"TensorRT Version: {trt.__version__}")
except ImportError:
    print("TensorRT: Not installed")

try:
    from torch2trt import torch2trt
    print("torch2trt: Installed")
except ImportError:
    print("torch2trt: Not installed")
```

## 💡 Önerilen Konfigürasyon

### Windows 10/11

- CUDA 11.8
- PyTorch 2.0+ with CUDA 11.8
- TensorRT 8.6+
- torch2trt (latest from GitHub)

### Linux (Ubuntu 20.04/22.04)

- CUDA 11.8 or 12.1
- PyTorch 2.0+ with CUDA
- TensorRT 8.6+
- torch2trt (latest from GitHub)

## 🎓 Alternatif: TensorRT Olmadan Kullanım

TensorRT kurulumu opsiyoneldir. TensorRT yüklü değilse:

```bash
# Normal PyTorch inference
python inference.py \
    --model_type mel_band_roformer \
    --config_path ckpts/config.yaml \
    --start_check_point ckpts/model.ckpt \
    --input_folder input/ \
    --store_dir output/
```

Proje otomatik olarak PyTorch inference'a geri döner.

## 📞 Yardım

Kurulum sorunları için:
1. GPU ve CUDA versiyonunuzu kontrol edin
2. PyTorch CUDA desteğini doğrulayın
3. TensorRT versiyonunun CUDA ile uyumlu olduğunu kontrol edin
4. GitHub Issues bölümünde arama yapın

---

**Son Güncelleme:** 2024
