# coding: utf-8
__author__ = 'TensorRT Inference Implementation'

import argparse
import time
import librosa
from tqdm.auto import tqdm
import sys
import os
import glob
import torch
import soundfile as sf
import torch.nn as nn
import numpy as np
from assets.i18n.i18n import I18nAuto

# Set inference path for compatibility
INFERENCE_PATH = os.path.abspath(__file__)

i18n = I18nAuto()

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils import get_model_from_config, normalize_audio, denormalize_audio
from utils import prefer_target_instrument, load_start_checkpoint
from tensorrt_backend import TensorRTBackend, is_tensorrt_available, load_model_from_checkpoint

import warnings
warnings.filterwarnings("ignore")


def shorten_filename(filename, max_length=30):
    """Dosya adını belirtilen maksimum uzunluğa kısaltır."""
    base, ext = os.path.splitext(filename)
    if len(base) <= max_length:
        return filename
    shortened = base[:15] + "..." + base[-10:] + ext
    return shortened


def get_soundfile_subtype(pcm_type, is_float=False):
    """PCM türüne göre uygun soundfile alt türünü belirler."""
    if is_float:
        return 'FLOAT'
    subtype_map = {
        'PCM_16': 'PCM_16',
        'PCM_24': 'PCM_24',
        'FLOAT': 'FLOAT'
    }
    return subtype_map.get(pcm_type, 'FLOAT')


def demix_tensorrt(
    config,
    trt_backend: TensorRTBackend,
    mix: np.ndarray,
    device: torch.device,
    pbar: bool = False
) -> dict:
    """
    TensorRT backend ile audio source separation.
    
    Parameters:
    ----------
    config : ConfigDict
        Configuration object
    trt_backend : TensorRTBackend
        TensorRT backend with compiled model
    mix : np.ndarray
        Input audio array
    device : torch.device
        Computation device
    pbar : bool
        Show progress bar
        
    Returns:
    -------
    dict
        Dictionary of separated sources
    """
    mix = torch.tensor(mix, dtype=torch.float32)
    
    chunk_size = config.audio.chunk_size
    num_instruments = len(prefer_target_instrument(config))
    num_overlap = config.inference.num_overlap
    
    fade_size = chunk_size // 10
    step = chunk_size // num_overlap
    border = chunk_size - step
    length_init = mix.shape[-1]
    
    # Windowing array
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    windowing_array = torch.ones(chunk_size)
    windowing_array[-fade_size:] = fadeout
    windowing_array[:fade_size] = fadein
    
    # Add padding
    if length_init > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode="reflect")
    
    batch_size = config.inference.batch_size
    use_amp = getattr(config.training, 'use_amp', True)
    
    with torch.cuda.amp.autocast(enabled=use_amp):
        with torch.inference_mode():
            # Initialize result and counter tensors
            req_shape = (num_instruments,) + mix.shape
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)
            
            i = 0
            batch_data = []
            batch_locations = []
            
            progress_bar = tqdm(
                total=mix.shape[1], 
                desc="Processing with TensorRT", 
                leave=False
            ) if pbar else None
            
            while i < mix.shape[1]:
                # Extract chunk
                part = mix[:, i:i + chunk_size].to(device)
                chunk_len = part.shape[-1]
                
                if chunk_len > chunk_size // 2:
                    pad_mode = "reflect"
                else:
                    pad_mode = "constant"
                    
                part = nn.functional.pad(
                    part, 
                    (0, chunk_size - chunk_len), 
                    mode=pad_mode, 
                    value=0
                )
                
                batch_data.append(part)
                batch_locations.append((i, chunk_len))
                i += step
                
                # Process batch
                if len(batch_data) >= batch_size or i >= mix.shape[1]:
                    arr = torch.stack(batch_data, dim=0)
                    
                    # Use TensorRT backend for inference
                    x = trt_backend(arr)
                    
                    window = windowing_array.clone()
                    if i - step == 0:  # First chunk
                        window[:fade_size] = 1
                    elif i >= mix.shape[1]:  # Last chunk
                        window[-fade_size:] = 1
                    
                    for j, (start, seg_len) in enumerate(batch_locations):
                        result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu() * window[..., :seg_len]
                        counter[..., start:start + seg_len] += window[..., :seg_len]
                    
                    batch_data.clear()
                    batch_locations.clear()
                
                if progress_bar:
                    progress_bar.update(step)
            
            if progress_bar:
                progress_bar.close()
            
            # Compute final estimated sources
            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)
            
            # Remove padding
            if length_init > 2 * border and border > 0:
                estimated_sources = estimated_sources[..., border:-border]
    
    # Return as dictionary
    instruments = prefer_target_instrument(config)
    ret_data = {k: v for k, v in zip(instruments, estimated_sources)}
    
    return ret_data


def run_folder_tensorrt(trt_backend, args, config, device, verbose: bool = False):
    """
    TensorRT backend ile klasör işleme.
    """
    start_time = time.time()
    
    mixture_paths = sorted(glob.glob(os.path.join(args.input_folder, '*.*')))
    sample_rate = getattr(config.audio, 'sample_rate', 44100)
    
    print(i18n("total_files_found").format(len(mixture_paths), sample_rate))
    print("🚀 Using TensorRT backend for acceleration")
    
    instruments = prefer_target_instrument(config)[:]
    
    # Çıktı klasörünü kullan
    store_dir = args.store_dir
    os.makedirs(store_dir, exist_ok=True)
    
    if not verbose:
        mixture_paths = tqdm(mixture_paths, desc=i18n("total_progress"))
    
    detailed_pbar = not args.disable_detailed_pbar
    print(i18n("detailed_pbar_enabled").format(detailed_pbar))
    
    for path in mixture_paths:
        try:
            mix, sr = librosa.load(path, sr=sample_rate, mono=False)
            print(i18n("loaded_audio").format(path, mix.shape))
        except Exception as e:
            print(i18n("cannot_read_track").format(path))
            print(i18n("error_message").format(str(e)))
            continue
        
        mix_orig = mix.copy()
        if 'normalize' in config.inference:
            if config.inference['normalize'] is True:
                mix, norm_params = normalize_audio(mix)
        
        # Use TensorRT backend
        waveforms_orig = demix_tensorrt(config, trt_backend, mix, device, pbar=detailed_pbar)
        
        if args.extract_instrumental:
            instr = 'vocals' if 'vocals' in instruments else instruments[0]
            waveforms_orig['instrumental'] = mix_orig - waveforms_orig[instr]
            if 'instrumental' not in instruments:
                instruments.append('instrumental')
        
        for instr in instruments:
            estimates = waveforms_orig[instr]
            if 'normalize' in config.inference:
                if config.inference['normalize'] is True:
                    estimates = denormalize_audio(estimates, norm_params)
            
            is_float = getattr(args, 'export_format', '').startswith('wav FLOAT')
            codec = 'flac' if getattr(args, 'flac_file', False) else 'wav'
            if codec == 'flac':
                subtype = get_soundfile_subtype(args.pcm_type, is_float)
            else:
                subtype = get_soundfile_subtype('FLOAT', is_float)
            
            shortened_filename = shorten_filename(os.path.basename(path))
            output_filename = f"{shortened_filename}_{instr}.{codec}"
            output_path = os.path.join(store_dir, output_filename)
            sf.write(output_path, estimates.T, sr, subtype=subtype)
    
    print(i18n("elapsed_time").format(time.time() - start_time))


def proc_folder_tensorrt(args):
    """
    TensorRT ile inference işleme fonksiyonu.
    """
    parser = argparse.ArgumentParser(description="TensorRT Inference for Music Source Separation")
    parser.add_argument("--model_type", type=str, default='mdx23c', help="Model type")
    parser.add_argument("--config_path", type=str, help="Config path")
    parser.add_argument("--start_check_point", type=str, default='', help="Checkpoint path (.ckpt)")
    parser.add_argument("--input_folder", type=str, help="Input folder path")
    parser.add_argument("--store_dir", type=str, default="", help="Output directory")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help="CUDA device IDs")
    parser.add_argument("--extract_instrumental", action='store_true', help="Extract instrumental")
    parser.add_argument("--disable_detailed_pbar", action='store_true', help="Disable detailed progress bar")
    parser.add_argument("--flac_file", action='store_true', help="Output as FLAC")
    parser.add_argument("--export_format", type=str, choices=['wav FLOAT', 'flac PCM_16', 'flac PCM_24'], 
                        default='flac PCM_24', help="Export format")
    parser.add_argument("--pcm_type", type=str, choices=['PCM_16', 'PCM_24'], default='PCM_24', help="PCM type")
    parser.add_argument("--chunk_size", type=int, default=1000000, help="Inference chunk size")
    parser.add_argument("--overlap", type=int, default=4, help="Inference overlap factor")
    parser.add_argument("--tensorrt_precision", type=str, choices=['fp32', 'fp16'], default='fp16',
                        help="TensorRT precision mode")
    parser.add_argument("--use_tensorrt_cache", action='store_true', help="Use TensorRT engine cache")
    parser.add_argument("--tensorrt_cache_dir", type=str, default='trt_cache', help="TensorRT cache directory")
    
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    
    # Check TensorRT availability
    if not is_tensorrt_available():
        print("❌ TensorRT is not available!")
        print("Please install with: pip install torch2trt")
        print("Falling back to standard inference...")
        from inference import proc_folder
        proc_folder(None)
        return
    
    # Device setup
    if torch.cuda.is_available():
        print(i18n("cuda_available"))
        device = f'cuda:{args.device_ids[0]}' if type(args.device_ids) == list else f'cuda:{args.device_ids}'
    else:
        print("❌ CUDA is required for TensorRT!")
        return
    
    print(i18n("using_device").format(device))
    
    # Load model
    model_load_start_time = time.time()
    torch.backends.cudnn.benchmark = True
    
    model, config = get_model_from_config(args.model_type, args.config_path)
    
    if args.start_check_point != '':
        # Load checkpoint
        model = load_model_from_checkpoint(args.start_check_point, model, device)
    
    print(i18n("instruments_print").format(config.training.instruments))
    
    # Compile to TensorRT
    print("🔧 Compiling model to TensorRT...")
    chunk_size = config.audio.chunk_size
    sample_rate = getattr(config.audio, 'sample_rate', 44100)
    
    # Create TensorRT backend
    trt_backend = TensorRTBackend(device=device, cache_dir=args.tensorrt_cache_dir)
    
    # Example input shape (batch_size, channels, chunk_size)
    batch_size = config.inference.batch_size
    num_channels = 2  # stereo
    example_input_shape = (batch_size, num_channels, chunk_size)
    
    # Compile model
    cache_key = f"{args.model_type}_{os.path.basename(args.start_check_point)}_{args.tensorrt_precision}"
    trt_backend.compile_model(
        model=model,
        input_shape=example_input_shape,
        precision=args.tensorrt_precision,
        use_cache=args.use_tensorrt_cache,
        cache_key=cache_key
    )
    
    print(i18n("model_load_time").format(time.time() - model_load_start_time))
    
    # Run inference
    run_folder_tensorrt(trt_backend, args, config, device, verbose=False)


if __name__ == "__main__":
    proc_folder_tensorrt(None)
