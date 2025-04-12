# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import os
import librosa
import soundfile as sf
import numpy as np
import argparse
import gc
import psutil
from scipy.signal import stft, istft
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()
import gc
import psutil

def stft(wave, nfft, hl):
    wave_left = np.asfortranarray(wave[0])
    wave_right = np.asfortranarray(wave[1])
    spec_left = librosa.stft(wave_left, n_fft=nfft, hop_length=hl)
    spec_right = librosa.stft(wave_right, n_fft=nfft, hop_length=hl)
    spec = np.asfortranarray([spec_left, spec_right])
    return spec


def istft(spec, hl, length):
    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])
    wave_left = librosa.istft(spec_left, hop_length=hl, length=length)
    wave_right = librosa.istft(spec_right, hop_length=hl, length=length)
    wave = np.asfortranarray([wave_left, wave_right])
    return wave


def absmax(a, *, axis):
    """Return the values of `a` where the absolute value is maximized along `axis`."""
    argmax = np.argmax(np.abs(a), axis=axis)
    if axis is not None:
        argmax = np.expand_dims(argmax, axis)
        result = np.take_along_axis(a, argmax, axis=axis)
        return np.squeeze(result, axis=axis)
    else:
        return a.flatten()[argmax]


def absmin(a, *, axis):
    """Return the values of `a` where the absolute value is minimized along `axis`."""
    argmin = np.argmin(np.abs(a), axis=axis)
    if axis is not None:
        argmin = np.expand_dims(argmin, axis)
        result = np.take_along_axis(a, argmin, axis=axis)
        return np.squeeze(result, axis=axis)
    else:
        return a.flatten()[argmin]


def lambda_max(arr, axis=None, key=None, keepdims=False):
    idxs = np.argmax(key(arr), axis)
    if axis is not None:
        idxs = np.expand_dims(idxs, axis)
        result = np.take_along_axis(arr, idxs, axis)
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return result
    else:
        return arr.flatten()[idxs]


def lambda_min(arr, axis=None, key=None, keepdims=False):
    idxs = np.argmin(key(arr), axis)
    if axis is not None:
        idxs = np.expand_dims(idxs, axis)
        result = np.take_along_axis(arr, idxs, axis)
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return result
    else:
        return arr.flatten()[idxs]


def average_waveforms(files, weights, algorithm, output_file, sr=48000):
    """Dosyaları akış tabanlı olarak okuyarak ensemble işlemini gerçekleştirir."""
    total_samples = int(librosa.get_duration(path=files[0]) * sr)
    weights = np.array(weights) / np.sum(weights) if weights is not None else np.ones(len(files)) / len(files)

    # Dosya kontrolü
    unique_files = list(dict.fromkeys([os.path.normpath(f) for f in files]))
    if len(unique_files) < len(files):
        print(f"Uyarı: Tekrar eden dosyalar tespit edildi, yalnızca benzersiz dosyalar işlenecek: {unique_files}")
        files = unique_files
        weights = np.array([weights[0]] * len(files)) / np.sum([weights[0]] * len(files))

    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Giriş dosyası eksik: {f}")
        if os.path.getsize(f) == 0:
            raise ValueError(f"Giriş dosyası boş: {f}")
        file_sr = librosa.get_samplerate(f)
        if file_sr != sr:
            print(f"Örnek oranı uyuşmazlığı: {f}, {file_sr}Hz, 48000 Hz’e yeniden örnekleniyor")
            audio, _ = librosa.load(f, sr=sr, mono=False)
            sf.write(f, audio.T, sr)

    # Bellek kontrolü
    available_memory = psutil.virtual_memory().available / (1024**3)  # GB
    print(f"Kullanılabilir bellek: {available_memory:.2f} GB")
    if available_memory < 4:
        raise MemoryError("Yetersiz bellek: Ensemble işlemi için en az 4 GB boş bellek gerekli.")

    if algorithm in ['avg_wave', 'median_wave']:
        # Hafif algoritmalar için akış tabanlı işleme
        with sf.SoundFile(output_file, 'w', sr, channels=2, subtype='FLOAT') as outfile:
            readers = [sf.SoundFile(f, 'r') for f in files]
            total_frames = readers[0].frames
            buffer_size = 16384  # 16384 örneklik buffer (~0.34 saniye @ 48000 Hz)

            for i in range(0, total_frames, buffer_size):
                end = min(i + buffer_size, total_frames)
                buffers = []
                for r in readers:
                    r.seek(i)
                    data = r.read(end-i)
                    buffers.append(data.T if data.shape[0] > 0 else np.zeros((2, end-i)))
                buffers = np.array(buffers)

                if algorithm == 'avg_wave':
                    weighted = [buffers[j] * weights[j] for j in range(len(buffers))]
                    result = np.sum(weighted, axis=0)
                else:  # median_wave
                    result = np.median(buffers, axis=0)

                outfile.write(result.T)

                # Belleği temizle
                del buffers
                del result
                gc.collect()

                print(f"İşlenen örnekler: {i}/{total_frames}, Bellek kullanımı: {psutil.virtual_memory().percent}%")

            for r in readers:
                r.close()

    else:
        # FFT tabanlı algoritmalar için spektrogram tabanlı işleme
        nfft = 1024  # Küçük FFT penceresi
        hop_length = 512  # Büyük adım boyutu
        buffer_size = nfft * 8  # FFT için yeterli buffer

        with sf.SoundFile(output_file, 'w', sr, channels=2, subtype='FLOAT') as outfile:
            readers = [sf.SoundFile(f, 'r') for f in files]
            total_frames = readers[0].frames

            for i in range(0, total_frames, buffer_size):
                end = min(i + buffer_size, total_frames)
                buffers = []
                for r in readers:
                    r.seek(i)
                    data = r.read(end-i)
                    buffers.append(data.T if data.shape[0] > 0 else np.zeros((2, end-i)))
                buffers = np.array(buffers)

                complex_spectrograms = []
                for b in buffers:
                    spec = []
                    for channel in range(b.shape[0]):
                        _, _, Zxx = stft(b[channel], nperseg=nfft, noverlap=nfft-hop_length)
                        spec.append(Zxx)
                    complex_spectrograms.append(np.array(spec))
                complex_spectrograms = np.array(complex_spectrograms)

                if algorithm == 'max_fft':
                    magnitude = np.max(np.abs(complex_spectrograms), axis=0)
                    phase = complex_spectrograms[np.argmax(np.abs(complex_spectrograms), axis=0)]
                elif algorithm == 'min_fft':
                    magnitude = np.min(np.abs(complex_spectrograms), axis=0)
                    phase = complex_spectrograms[np.argmin(np.abs(complex_spectrograms), axis=0)]
                elif algorithm == 'median_fft':
                    magnitude = np.median(np.abs(complex_spectrograms), axis=0)
                    phase = complex_spectrograms[np.argsort(np.abs(complex_spectrograms), axis=0)[len(complex_spectrograms)//2]]

                complex_spectrogram = magnitude * np.exp(1j * np.angle(phase))
                result = np.zeros((complex_spectrogram.shape[0], end-i))
                for channel in range(complex_spectrogram.shape[0]):
                    _, result[channel] = istft(complex_spectrogram[channel], nperseg=nfft, noverlap=nfft-hop_length)
                    if result[channel].shape[0] > (end-i):
                        result[channel] = result[channel][:(end-i)]
                    elif result[channel].shape[0] < (end-i):
                        result[channel] = np.pad(result[channel], (0, (end-i) - result[channel].shape[0]), 'constant')

                outfile.write(result.T)

                # Belleği temizle
                del buffers
                del complex_spectrograms
                del magnitude
                del phase
                del result
                gc.collect()

                print(f"İşlenen örnekler: {i}/{total_frames}, Bellek kullanımı: {psutil.virtual_memory().percent}%")

            for r in readers:
                r.close()

def ensemble_files(args):
    parser = argparse.ArgumentParser(description=i18n("ensemble_files_description"))
    parser.add_argument("--files", type=str, required=True, nargs='+', help=i18n("ensemble_files_help"))
    parser.add_argument("--type", type=str, default='avg_wave', choices=['avg_wave', 'median_wave', 'max_fft', 'min_fft', 'median_fft'], help=i18n("ensemble_type_help"))
    parser.add_argument("--weights", type=float, nargs='+', help=i18n("ensemble_weights_help"))
    parser.add_argument("--output", default="res.wav", type=str, help=i18n("ensemble_output_help"))
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    print(i18n("ensemble_type_print").format(args.type))
    print(i18n("num_input_files_print").format(len(args.files)))
    weights = args.weights if args.weights is not None else [1.0] * len(args.files)
    print(i18n("weights_print").format(weights))
    print(i18n("output_file_print").format(args.output))

    # Dosya sürelerini kontrol et
    durations = []
    for f in args.files:
        try:
            durations.append(librosa.get_duration(path=f))
        except Exception as e:
            raise FileNotFoundError(f"Dosya süresi alınamadı: {f}, hata: {str(e)}")
    if not all(abs(d - durations[0]) < 0.01 for d in durations):
        raise ValueError(i18n("duration_mismatch_error"))

    average_waveforms(args.files, weights, args.type, args.output)

    print(i18n("ensemble_completed_print").format(args.output))

if __name__ == "__main__":
    ensemble_files(None)
