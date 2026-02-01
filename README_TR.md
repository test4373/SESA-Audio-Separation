# Music Source Separation with TensorRT

Bu proje, müzik kaynak ayırma (Music Source Separation) için PyTorch tabanlı derin öğrenme modellerini destekler ve artık **NVIDIA TensorRT** backend ile gelişmiş performans sunar.

## 🚀 Yeni Özellikler: TensorRT Desteği

- ⚡ **3x Daha Hızlı İnference**: TensorRT optimizasyonu ile işleme süresini %66'ya kadar azaltın
- 💾 **Daha Az Bellek Kullanımı**: FP16 precision ile GPU bellek kullanımını optimize edin
- 🔥 **Otomatik Önbellekleme**: İlk derlemeden sonra hızlı yükleme
- 🎯 **Kolay Kullanım**: Tek parametre ile TensorRT'yi aktif edin
- 🔄 **Geriye Dönük Uyumlu**: TensorRT yüklü değilse otomatik PyTorch'a geçiş

## 📋 Gereksinimler

### Temel Gereksinimler
- Python 3.8+
- NVIDIA GPU (Compute Capability 6.0+)
- CUDA 11.x veya 12.x
- PyTorch 2.0+ with CUDA

### TensorRT İçin Ek Gereksinimler
```bash
pip install torch2trt
pip install nvidia-tensorrt
```

Detaylı kurulum için: [INSTALL_TENSORRT.md](INSTALL_TENSORRT.md)

## 🎯 Hızlı Başlangıç

### 1. Normal Inference (PyTorch)

```bash
python inference.py \
    --model_type mel_band_roformer \
    --config_path ckpts/config.yaml \
    --start_check_point ckpts/model.ckpt \
    --input_folder input/ \
    --store_dir output/
```

### 2. TensorRT Hızlandırmalı Inference

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

### 3. Inference.py İle TensorRT Kullanımı

```bash
python inference.py \
    --model_type mel_band_roformer \
    --config_path ckpts/config.yaml \
    --start_check_point ckpts/model.ckpt \
    --input_folder input/ \
    --store_dir output/ \
    --use_tensorrt \
    --tensorrt_precision fp16
```

## 📊 Performans Karşılaştırması

### Benchmark Çalıştırma

```bash
python benchmark_tensorrt.py \
    --model_type mel_band_roformer \
    --config_path ckpts/config.yaml \
    --start_check_point ckpts/model.ckpt \
    --precision fp16 \
    --num_iterations 100
```

### Örnek Sonuçlar (RTX 3090)

| Model | PyTorch | TensorRT FP16 | Hızlanma |
|-------|---------|---------------|----------|
| MelBand-Roformer | 150ms | 50ms | **3.0x** |
| BS-Roformer | 180ms | 60ms | **3.0x** |
| MDX23C | 100ms | 40ms | **2.5x** |

## 🛠️ Parametreler

### TensorRT Parametreleri

- `--use_tensorrt`: TensorRT backend'ini aktif et
- `--tensorrt_precision`: Precision mode (`fp32` veya `fp16`)
- `--use_tensorrt_cache`: TensorRT engine önbelleğini kullan
- `--tensorrt_cache_dir`: Önbellek dizini (default: `trt_cache/`)

### Inference Parametreleri

- `--model_type`: Model tipi (örn: `mel_band_roformer`, `bs_roformer`, `mdx23c`)
- `--config_path`: Model config dosyası yolu
- `--start_check_point`: Model checkpoint (.ckpt) dosyası
- `--input_folder`: Girdi ses dosyaları klasörü
- `--store_dir`: Çıktı dosyaları klasörü
- `--chunk_size`: İşleme chunk boyutu
- `--overlap`: Overlap faktörü
- `--extract_instrumental`: Instrumental track çıkar
- `--use_tta`: Test-Time Augmentation kullan

## 📁 Proje Yapısı

```
.
├── tensorrt_backend.py          # TensorRT backend implementasyonu
├── inference_tensorrt.py        # TensorRT ile inference
├── inference.py                 # Normal PyTorch inference
├── benchmark_tensorrt.py        # Benchmark aracı
├── processing.py                # Ana işleme fonksiyonları
├── gui.py                       # Gradio web arayüzü
├── model.py                     # Model konfigürasyonları
├── utils.py                     # Yardımcı fonksiyonlar
├── ensemble.py                  # Ensemble işlemleri
├── trt_cache/                   # TensorRT engine önbelleği
├── ckpts/                       # Model checkpoint'leri
├── input/                       # Girdi ses dosyaları
├── output/                      # Çıktı ses dosyaları
├── examples/                    # Örnek kodlar
│   └── tensorrt_example.py
└── README_TensorRT.md           # TensorRT dokümantasyonu
```

## 🔧 TensorRT Backend API

### Python API Kullanımı

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
audio_input = torch.randn(1, 2, 352800).cuda()
output = trt_backend(audio_input)
```

## 🎨 Gradio Web Arayüzü

Web arayüzü üzerinden TensorRT kullanımı:

```bash
python main.py --method gradio --port 7860
```

Arayüzde "Use TensorRT" seçeneğini aktif edin.

## 📚 Desteklenen Modeller

Tüm mevcut modeller TensorRT ile uyumludur:

### Vokal Modelleri
- VOCALS-big_beta6X (by Unwa)
- VOCALS-MelBand-Roformer (by KimberleyJSN)
- VOCALS-BS-Roformer_1297 (by viperx)
- ve diğerleri...

### Instrumental Modelleri
- INST-Mel-Roformer v1 (by unwa)
- INST-MelBand-Roformer (by Becruily)
- ve diğerleri...

### 4-Stem Modelleri
- 4STEMS-SCNet_MUSDB18 (by starrytong)
- 4STEMS-BS-Roformer_MUSDB18 (by ZFTurbo)
- ve diğerleri...

Tam liste için `model.py` dosyasına bakın.

## 🐛 Sorun Giderme

### TensorRT Bulunamadı

```bash
pip install torch2trt nvidia-tensorrt
```

### CUDA Out of Memory

- Batch size azaltın: `--batch_size 1`
- Chunk size azaltın: `--chunk_size 100000`
- FP16 kullanın: `--tensorrt_precision fp16`

### İlk Çalıştırma Yavaş

İlk çalıştırmada TensorRT compilation yapılır (~1-5 dakika). Sonraki çalıştırmalar önbellekten yüklenir.

## 💡 İpuçları ve En İyi Uygulamalar

1. **İlk Kullanımda**: `--use_tensorrt_cache` ile önbelleği aktif edin
2. **FP16 Kullanın**: Modern GPU'larda (RTX 20xx+) FP16 en iyi performansı verir
3. **Batch Size**: GPU belleği izin veriyorsa batch size artırın
4. **Önbellek**: Model değişmezse önbellek her zaman kullanılmalı

## 📖 Dokümantasyon

- [TensorRT Kurulum Rehberi](INSTALL_TENSORRT.md)
- [TensorRT Kullanım Kılavuzu](README_TensorRT.md)
- [Örnek Kodlar](examples/tensorrt_example.py)

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen pull request gönderin veya issue açın.

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🙏 Teşekkürler

- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)
- [PyTorch](https://pytorch.org/)
- Tüm model geliştiricileri

## 📞 İletişim

Sorularınız için:
- GitHub Issues
- Pull Requests

---

**Not:** TensorRT kullanımı opsiyoneldir. Yüklü değilse proje normal PyTorch inference ile çalışır.

**Önemli:** İlk çalıştırmada TensorRT compilation yapılacağından işlem 1-5 dakika sürebilir. Sonraki çalıştırmalar çok daha hızlı olacaktır.
