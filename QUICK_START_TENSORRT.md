# TensorRT Hızlı Başlangıç Rehberi

Bu rehber, projedeki TensorRT backend'ini hızlıca kullanmaya başlamanız için hazırlanmıştır.

## ⚡ 5 Dakikada TensorRT

### Adım 1: Kurulum

```bash
# TensorRT paketlerini yükle
pip install torch2trt nvidia-tensorrt

# Veya tüm gereksinimleri yükle
pip install -r requirements.txt
```

### Adım 2: İlk Çalıştırma

```bash
# Input klasörüne ses dosyanızı koyun
cp your_audio.wav input/

# TensorRT ile işleyin
python inference_tensorrt.py \
    --model_type mel_band_roformer \
    --config_path ckpts/config.yaml \
    --start_check_point ckpts/model.ckpt \
    --input_folder input/ \
    --store_dir output/ \
    --tensorrt_precision fp16 \
    --use_tensorrt_cache
```

### Adım 3: Çıktıyı Kontrol Edin

```bash
ls output/
# Çıktı: your_audio_Vocals_ModelName.wav
#        your_audio_Instrumental_ModelName.wav
```

## 🎯 Hızlı Komutlar

### Temel Kullanım

```bash
# FP16 precision ile (önerilen)
python inference_tensorrt.py \
    --model_type mel_band_roformer \
    --config_path ckpts/config.yaml \
    --start_check_point ckpts/model.ckpt \
    --input_folder input/ \
    --store_dir output/ \
    --tensorrt_precision fp16 \
    --use_tensorrt_cache
```

### FP32 Precision

```bash
python inference_tensorrt.py \
    --model_type mel_band_roformer \
    --config_path ckpts/config.yaml \
    --start_check_point ckpts/model.ckpt \
    --input_folder input/ \
    --store_dir output/ \
    --tensorrt_precision fp32
```

### Instrumental Çıkart

```bash
python inference_tensorrt.py \
    --model_type mel_band_roformer \
    --config_path ckpts/config.yaml \
    --start_check_point ckpts/model.ckpt \
    --input_folder input/ \
    --store_dir output/ \
    --extract_instrumental \
    --tensorrt_precision fp16
```

## 📊 Benchmark

### Hızlı Performans Testi

```bash
python benchmark_tensorrt.py \
    --model_type mel_band_roformer \
    --config_path ckpts/config.yaml \
    --start_check_point ckpts/model.ckpt \
    --num_iterations 50
```

### Detaylı Benchmark

```bash
python benchmark_tensorrt.py \
    --model_type mel_band_roformer \
    --config_path ckpts/config.yaml \
    --start_check_point ckpts/model.ckpt \
    --precision fp16 \
    --num_iterations 100 \
    --warmup_iterations 20
```

## 🔥 Popüler Model Örnekleri

### Vokal Ayırma (MelBand-Roformer)

```bash
python inference_tensorrt.py \
    --model_type mel_band_roformer \
    --config_path ckpts/big_beta6x.yaml \
    --start_check_point ckpts/big_beta6x.ckpt \
    --input_folder input/ \
    --store_dir output/ \
    --tensorrt_precision fp16 \
    --use_tensorrt_cache
```

### Vokal Ayırma (BS-Roformer)

```bash
python inference_tensorrt.py \
    --model_type bs_roformer \
    --config_path ckpts/model_bs_roformer_ep_317_sdr_12.9755.yaml \
    --start_check_point ckpts/model_bs_roformer_ep_317_sdr_12.9755.ckpt \
    --input_folder input/ \
    --store_dir output/ \
    --tensorrt_precision fp16
```

### 4-Stem Ayırma (SCNet)

```bash
python inference_tensorrt.py \
    --model_type scnet \
    --config_path ckpts/config_musdb18_scnet.yaml \
    --start_check_point ckpts/scnet_checkpoint_musdb18.ckpt \
    --input_folder input/ \
    --store_dir output/ \
    --tensorrt_precision fp16
```

## 🎨 Gradio Web UI

### TensorRT ile Web UI

```bash
# Web UI'yi başlat
python main.py --method gradio --port 7860

# Tarayıcıda aç: http://localhost:7860
# "Use TensorRT" checkbox'ını işaretle
```

## 💻 Python API Kullanımı

### Basit Örnek

```python
from tensorrt_backend import TensorRTBackend, load_model_from_checkpoint
from utils import get_model_from_config
import torch

# Model yükle
model, config = get_model_from_config('mel_band_roformer', 'ckpts/config.yaml')
model = load_model_from_checkpoint('ckpts/model.ckpt', model, 'cuda:0')

# TensorRT backend
trt_backend = TensorRTBackend(device='cuda:0')
trt_backend.compile_model(
    model=model,
    input_shape=(1, 2, 352800),
    precision='fp16',
    use_cache=True
)

# Inference
audio = torch.randn(1, 2, 352800).cuda()
output = trt_backend(audio)
```

### Detaylı Örnekler

```bash
# Örnekleri çalıştır
python examples/tensorrt_example.py
```

## 📋 Model İndirme

### Otomatik İndirme

Modeller ilk kullanımda otomatik olarak indirilir:

```bash
python inference_tensorrt.py \
    --model_type mel_band_roformer \
    --config_path ckpts/big_beta6x.yaml \
    --start_check_point ckpts/big_beta6x.ckpt \
    --input_folder input/ \
    --store_dir output/
```

### Manuel İndirme

```python
from model import download_file

# Model config
download_file('https://huggingface.co/.../config.yaml')

# Model checkpoint
download_file('https://huggingface.co/.../model.ckpt')
```

## ⚙️ Konfigürasyon

### Chunk Size ve Overlap

```bash
# Büyük chunk size (daha hızlı ama daha fazla memory)
python inference_tensorrt.py \
    --model_type mel_band_roformer \
    --config_path ckpts/config.yaml \
    --start_check_point ckpts/model.ckpt \
    --input_folder input/ \
    --store_dir output/ \
    --chunk_size 500000 \
    --overlap 2

# Küçük chunk size (daha yavaş ama daha az memory)
python inference_tensorrt.py \
    --model_type mel_band_roformer \
    --config_path ckpts/config.yaml \
    --start_check_point ckpts/model.ckpt \
    --input_folder input/ \
    --store_dir output/ \
    --chunk_size 200000 \
    --overlap 4
```

### Batch Processing

```bash
# Birden fazla dosya işle
cp song1.wav input/
cp song2.wav input/
cp song3.wav input/

python inference_tensorrt.py \
    --model_type mel_band_roformer \
    --config_path ckpts/config.yaml \
    --start_check_point ckpts/model.ckpt \
    --input_folder input/ \
    --store_dir output/
```

## 🔍 Sorun Giderme

### TensorRT Yüklenmedi

```bash
pip install torch2trt nvidia-tensorrt --force-reinstall
```

### CUDA Hatası

```bash
# CUDA versiyonunu kontrol et
python -c "import torch; print(torch.version.cuda)"

# Uyumlu PyTorch yükle
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Memory Hatası

```bash
# Chunk size'ı azalt
python inference_tensorrt.py ... --chunk_size 100000

# FP16 kullan
python inference_tensorrt.py ... --tensorrt_precision fp16
```

## 📈 Performans İpuçları

### En Hızlı Ayarlar

```bash
python inference_tensorrt.py \
    --model_type mel_band_roformer \
    --config_path ckpts/config.yaml \
    --start_check_point ckpts/model.ckpt \
    --input_folder input/ \
    --store_dir output/ \
    --tensorrt_precision fp16 \
    --use_tensorrt_cache \
    --chunk_size 500000 \
    --overlap 2
```

### En Yüksek Kalite Ayarlar

```bash
python inference_tensorrt.py \
    --model_type mel_band_roformer \
    --config_path ckpts/config.yaml \
    --start_check_point ckpts/model.ckpt \
    --input_folder input/ \
    --store_dir output/ \
    --tensorrt_precision fp32 \
    --chunk_size 300000 \
    --overlap 8 \
    --use_tta
```

## 🎓 Daha Fazla Bilgi

- [Detaylı TensorRT Dokümantasyonu](README_TensorRT.md)
- [Kurulum Rehberi](INSTALL_TENSORRT.md)
- [Örnek Kodlar](examples/tensorrt_example.py)
- [Ana Dokümantasyon](README_TR.md)

## 💬 Yardım

Sorun yaşıyorsanız:

1. `benchmark_tensorrt.py` ile test edin
2. [INSTALL_TENSORRT.md](INSTALL_TENSORRT.md) kurulum rehberini takip edin
3. GitHub Issues'da arama yapın
4. Yeni issue açın

---

**İpucu:** İlk çalıştırmada TensorRT compilation ~1-5 dakika sürer. Sonraki çalıştırmalar önbellekten yüklenir ve çok hızlıdır!

**Önemli:** `--use_tensorrt_cache` parametresini kullanmayı unutmayın!
