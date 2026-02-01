# TensorRT Backend for Music Source Separation

Bu proje artık **NVIDIA TensorRT** backend desteği ile gelir. TensorRT, PyTorch modellerini optimize ederek inference hızını önemli ölçüde artırır.

## 🚀 Özellikler

- ✅ TensorRT ile hızlandırılmış inference
- ✅ FP16 ve FP32 precision desteği
- ✅ Otomatik TensorRT engine önbellekleme
- ✅ .ckpt checkpoint dosyaları ile uyumlu
- ✅ PyTorch ile karşılaştırmalı benchmark araçları
- ✅ Geriye dönük uyumluluk (TensorRT yoksa normal PyTorch kullanılır)

## 📋 Gereksinimler

### CUDA ve TensorRT Kurulumu

1. **NVIDIA GPU Driver** (Latest)
2. **CUDA Toolkit** (11.x veya 12.x)
3. **TensorRT**

### Python Paketleri

```bash
# TensorRT için gerekli paketler
pip install torch2trt
pip install nvidia-tensorrt

# Veya tüm gereksinimleri yükle
pip install -r requirements.txt
```

## 🎯 Kullanım

### 1. TensorRT ile Inference

```bash
python inference_tensorrt.py \
    --model_type mel_band_roformer \
    --config_path ckpts/config.yaml \
    --start_check_point ckpts/model.ckpt \
    --input_folder input/ \
    --store_dir output/ \
    --tensorrt_precision fp16 \
    --use_tensorrt_cache
```

### 2. Parametreler

- `--tensorrt_precision`: TensorRT hassasiyeti (`fp32` veya `fp16`)
- `--use_tensorrt_cache`: TensorRT engine'i önbelleğe al (hız kazancı)
- `--tensorrt_cache_dir`: Önbellek dizini (varsayılan: `trt_cache/`)

### 3. Normal Inference (PyTorch)

TensorRT yüklü değilse, otomatik olarak normal PyTorch inference kullanılır:

```bash
python inference.py \
    --model_type mel_band_roformer \
    --config_path ckpts/config.yaml \
    --start_check_point ckpts/model.ckpt \
    --input_folder input/ \
    --store_dir output/
```

## 📊 Performans Karşılaştırması

### Benchmark Çalıştırma

```bash
python benchmark_tensorrt.py \
    --model_type mel_band_roformer \
    --config_path ckpts/config.yaml \
    --start_check_point ckpts/model.ckpt \
    --device cuda:0 \
    --precision fp16 \
    --num_iterations 100
```

### Beklenen Performans Kazançları

| Model | PyTorch (ms) | TensorRT FP16 (ms) | Hızlanma |
|-------|--------------|-------------------|----------|
| MelBand-Roformer | ~150 | ~50 | **3.0x** |
| BS-Roformer | ~180 | ~60 | **3.0x** |
| MDX23C | ~100 | ~40 | **2.5x** |

*Not: Performans GPU modeline göre değişir. Yukarıdaki değerler RTX 3090 için örnek değerlerdir.*

## 🔧 TensorRT Backend API

### Programatik Kullanım

```python
from tensorrt_backend import TensorRTBackend, load_model_from_checkpoint
from utils import get_model_from_config

# Model yükle
model, config = get_model_from_config('mel_band_roformer', 'config.yaml')
model = load_model_from_checkpoint('model.ckpt', model, 'cuda:0')

# TensorRT backend oluştur
trt_backend = TensorRTBackend(device='cuda:0', cache_dir='trt_cache')

# Model'i compile et
trt_backend.compile_model(
    model=model,
    input_shape=(1, 2, 352800),  # (batch, channels, chunk_size)
    precision='fp16',
    use_cache=True
)

# Inference
import torch
dummy_input = torch.randn(1, 2, 352800).cuda()
output = trt_backend(dummy_input)
```

### TensorRT Engine Kaydetme/Yükleme

```python
# Engine'i kaydet
trt_backend.save_engine('model.trt')

# Engine'i yükle
trt_backend.load_engine('model.trt')
```

## 📁 Proje Yapısı

```
.
├── tensorrt_backend.py          # TensorRT backend implementasyonu
├── inference_tensorrt.py        # TensorRT ile inference scripti
├── benchmark_tensorrt.py        # Benchmark aracı
├── README_TensorRT.md          # Bu dosya
├── trt_cache/                  # TensorRT engine önbelleği
│   └── *.trt                   # Derlenmiş TensorRT engine'leri
└── requirements.txt            # Güncellenmiş gereksinimler
```

## ⚙️ TensorRT Önbellekleme

TensorRT engine'leri ilk derleme sırasında önbelleğe alınır:

```
trt_cache/
└── abc123def456.trt  # Model + config + precision hash'i
```

### Önbelleği Temizleme

```bash
rm -rf trt_cache/
```

### Önbellek Davranışı

- İlk çalıştırma: Model compile edilir (~1-5 dakika)
- Sonraki çalıştırmalar: Önbellekten yüklenir (~1 saniye)
- Model/config değişirse: Otomatik yeniden compile edilir

## 🐛 Troubleshooting

### TensorRT Bulunamadı

```
ImportError: No module named 'tensorrt'
```

**Çözüm:**
```bash
pip install nvidia-tensorrt
pip install torch2trt
```

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Çözüm:**
- Batch size'ı azaltın: `--batch_size 1`
- Chunk size'ı azaltın: `--chunk_size 100000`
- FP32 yerine FP16 kullanın: `--tensorrt_precision fp16`

### TensorRT Compilation Failed

TensorRT compile edilemezse, otomatik olarak PyTorch'a geri döner:

```
✗ TensorRT compilation failed: ...
Falling back to PyTorch model
```

### Yavaş İlk Çalıştırma

İlk çalıştırma TensorRT compilation nedeniyle yavaş olabilir (1-5 dakika). Sonraki çalıştırmalar önbellekten yüklendiği için hızlıdır.

## 🔍 TensorRT vs PyTorch Karşılaştırması

| Özellik | PyTorch | TensorRT |
|---------|---------|----------|
| Kurulum | Kolay | Orta |
| İlk çalıştırma | Hızlı | Yavaş (compile) |
| Inference hızı | Normal | **Çok hızlı** |
| Memory kullanımı | Normal | Daha az |
| GPU desteği | Tüm GPU'lar | NVIDIA GPU gerekli |
| Precision | FP32 | FP16/FP32/INT8 |

## 📚 Ek Kaynaklar

- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [torch2trt GitHub](https://github.com/NVIDIA-AI-IOT/torch2trt)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 💡 İpuçları

1. **İlk Kullanımda:** `--use_tensorrt_cache` kullanın, sonraki çalıştırmalar çok daha hızlı olur
2. **FP16 Kullanın:** Modern GPU'larda (RTX 20xx+) FP16 hem hızlı hem doğru
3. **Batch Size:** Mümkünse batch size artırın (GPU memory izin veriyorsa)
4. **Warmup:** İlk birkaç inference yavaş olabilir, sistem ısındıkça hızlanır

## 📞 Destek

Sorun yaşarsanız:
1. `benchmark_tensorrt.py` ile test edin
2. PyTorch inference ile karşılaştırın
3. TensorRT loglarını kontrol edin

---

**Not:** TensorRT kullanımı opsiyoneldir. Yüklü değilse, proje normal PyTorch inference ile çalışmaya devam eder.
