# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import os
import librosa
import soundfile as sf
import numpy as np
import argparse
import gc

def stft(wave, nfft, hl):
    wave_left = np.asfortranarray(wave[0])
    wave_right = np.asfortranarray(wave[1])
    spec_left = librosa.stft(wave_left, n_fft=nfft, hop_length=hl, window='hann')
    spec_right = librosa.stft(wave_right, n_fft=nfft, hop_length=hl, window='hann')
    spec = np.asfortranarray([spec_left, spec_right])
    return spec

def istft(spec, hl, length):
    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])
    wave_left = librosa.istft(spec_left, hop_length=hl, length=length, window='hann')
    wave_right = librosa.istft(spec_right, hop_length=hl, length=length, window='hann')
    wave = np.asfortranarray([wave_left, wave_right])
    return wave

def absmax(a, *, axis):
    dims = list(a.shape)
    dims.pop(axis)
    indices = list(np.ogrid[tuple(slice(0, d) for d in dims)])
    argmax = np.abs(a).argmax(axis=axis)
    insert_pos = (len(a.shape) + axis) % len(a.shape)
    indices.insert(insert_pos, argmax)
    return a[tuple(indices)]

def absmin(a, *, axis):
    dims = list(a.shape)
    dims.pop(axis)
    indices = list(np.ogrid[tuple(slice(0, d) for d in dims)])
    argmax = np.abs(a).argmin(axis=axis)
    insert_pos = (len(a.shape) + axis) % len(a.shape)
    indices.insert(insert_pos, argmax)
    return a[tuple(indices)]

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

def average_waveforms(pred_track, weights, algorithm, chunk_length):
    pred_track = np.array(pred_track)
    pred_track = np.array([p[:, :chunk_length] if p.shape[1] > chunk_length else np.pad(p, ((0, 0), (0, chunk_length - p.shape[1])), 'constant') for p in pred_track])
    mod_track = []
    
    for i in range(pred_track.shape[0]):
        if algorithm == 'avg_wave':
            mod_track.append(pred_track[i] * weights[i])
        elif algorithm in ['median_wave', 'min_wave', 'max_wave']:
            mod_track.append(pred_track[i])
        elif algorithm in ['avg_fft', 'min_fft', 'max_fft', 'median_fft']:
            spec = stft(pred_track[i], nfft=2048, hl=1024)
            if algorithm == 'avg_fft':
                mod_track.append(spec * weights[i])
            else:
                mod_track.append(spec)
    pred_track = np.array(mod_track)

    if algorithm == 'avg_wave':
        pred_track = pred_track.sum(axis=0)
        pred_track /= np.array(weights).sum()
    elif algorithm == 'median_wave':
        pred_track = np.median(pred_track, axis=0)
    elif algorithm == 'min_wave':
        pred_track = lambda_min(pred_track, axis=0, key=np.abs)
    elif algorithm == 'max_wave':
        pred_track = lambda_max(pred_track, axis=0, key=np.abs)
    elif algorithm == 'avg_fft':
        pred_track = pred_track.sum(axis=0)
        pred_track /= np.array(weights).sum()
        pred_track = istft(pred_track, 1024, chunk_length)
    elif algorithm == 'min_fft':
        pred_track = lambda_min(pred_track, axis=0, key=np.abs)
        pred_track = istft(pred_track, 1024, chunk_length)
    elif algorithm == 'max_fft':
        pred_track = absmax(pred_track, axis=0)
        pred_track = istft(pred_track, 1024, chunk_length)
    elif algorithm == 'median_fft':
        pred_track = np.median(pred_track, axis=0)
        pred_track = istft(pred_track, 1024, chunk_length)

    return pred_track

def ensemble_files(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=str, required=True, nargs='+', help="Path to all audio-files to ensemble")
    parser.add_argument("--type", type=str, default='avg_wave', help="One of avg_wave, median_wave, min_wave, max_wave, avg_fft, median_fft, min_fft, max_fft")
    parser.add_argument("--weights", type=float, nargs='+', help="Weights to create ensemble. Number of weights must be equal to number of files")
    parser.add_argument("--output", default="res.wav", type=str, help="Path to wav file where ensemble result will be stored")
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    print('Ensemble type: {}'.format(args.type))
    print('Number of input files: {}'.format(len(args.files)))
    if args.weights is not None:
        weights = np.array(args.weights)
    else:
        weights = np.ones(len(args.files))
    print('Weights: {}'.format(weights))
    print('Output file: {}'.format(args.output))

    durations = [librosa.get_duration(filename=f) for f in args.files]
    if not all(d == durations[0] for d in durations):
        raise ValueError("All files must have the same duration")

    total_duration = durations[0]
    sr = librosa.get_samplerate(args.files[0])
    chunk_duration = 30  # 30 saniyelik parçalar
    overlap_duration = 0.1  # 100 ms overlap
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap_duration * sr)
    step_samples = chunk_samples - overlap_samples  # Adım boyutu overlap ile azaltılır
    total_samples = int(total_duration * sr)

    # Chunk uzunluğunu hop_length ile hizala
    hop_length = 1024
    chunk_samples = ((chunk_samples + hop_length - 1) // hop_length) * hop_length
    step_samples = chunk_samples - overlap_samples

    with sf.SoundFile(args.output, 'w', sr, channels=2, subtype='FLOAT') as outfile:
        for start in range(0, total_samples, step_samples):
            end = min(start + chunk_samples, total_samples)
            chunk_length = end - start
            data = []

            for f in args.files:
                if not os.path.isfile(f):
                    print('Error. Can\'t find file: {}. Check paths.'.format(f))
                    exit()
                print(f'Reading chunk from file: {f} (start: {start/sr}s, duration: {(end-start)/sr}s)')
                wav, _ = librosa.load(f, sr=sr, mono=False, offset=start/sr, duration=(end-start)/sr)
                data.append(wav)

            res = average_waveforms(data, weights, args.type, chunk_length)
            res = res.astype(np.float32)
            print(f'Chunk result shape: {res.shape}')

            # Crossfade ile birleştirme
            if start > 0:
                outfile.seek(outfile.tell() - overlap_samples)
                prev_data = outfile.read(overlap_samples).T
                new_data = res[:, :overlap_samples]
                fade_out = np.linspace(1, 0, overlap_samples)
                fade_in = np.linspace(0, 1, overlap_samples)
                blended = prev_data * fade_out + new_data * fade_in
                outfile.seek(outfile.tell() - overlap_samples)
                outfile.write(blended.T)
                outfile.write(res[:, overlap_samples:].T)
            else:
                outfile.write(res.T)

            del data
            del res
            gc.collect()

    print(f'Ensemble completed. Output saved to: {args.output}')

if __name__ == "__main__":
    ensemble_files(None)
