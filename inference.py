# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

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

# Colab kontrolü
try:
    from google.colab import drive
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

i18n = I18nAuto()

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils import demix, get_model_from_config, normalize_audio, denormalize_audio
from utils import prefer_target_instrument, apply_tta, load_start_checkpoint, load_lora_weights

# PyTorch optimized backend (always available)
try:
    from pytorch_backend import PyTorchBackend
    PYTORCH_OPTIMIZED_AVAILABLE = True
except ImportError:
    PYTORCH_OPTIMIZED_AVAILABLE = False

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

def run_folder(model, args, config, device, verbose: bool = False):
    start_time = time.time()
    model.eval()

    mixture_paths = sorted(glob.glob(os.path.join(args.input_folder, '*.*')))
    sample_rate = getattr(config.audio, 'sample_rate', 44100)

    print(i18n("total_files_found").format(len(mixture_paths), sample_rate))

    instruments = prefer_target_instrument(config)[:]

    # Çıktı klasörünü kullan (processing.py tarafından ayarlandı)
    store_dir = args.store_dir
    os.makedirs(store_dir, exist_ok=True)

    if not verbose:
        mixture_paths = tqdm(mixture_paths, desc=i18n("total_progress"))
    else:
        mixture_paths = mixture_paths

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

        waveforms_orig = demix(config, model, mix, device, model_type=args.model_type, pbar=detailed_pbar)

        if args.use_tta:
            waveforms_orig = apply_tta(config, model, mix, waveforms_orig, device, args.model_type)

        if args.demud_phaseremix_inst:
            print(i18n("demudding_track").format(path))
            instr = 'vocals' if 'vocals' in instruments else instruments[0]
            instruments.append('instrumental_phaseremix')
            if 'instrumental' not in instruments and 'Instrumental' not in instruments:
                mix_modified = mix_orig - 2*waveforms_orig[instr]
                mix_modified_ = mix_modified.copy()
                waveforms_modified = demix(config, model, mix_modified, device, model_type=args.model_type, pbar=detailed_pbar)
                if args.use_tta:
                    waveforms_modified = apply_tta(config, model, mix_modified, waveforms_modified, device, args.model_type)
                waveforms_orig['instrumental_phaseremix'] = mix_orig + waveforms_modified[instr]
            else:
                mix_modified = 2*waveforms_orig[instr] - mix_orig
                mix_modified_ = mix_modified.copy()
                waveforms_modified = demix(config, model, mix_modified, device, model_type=args.model_type, pbar=detailed_pbar)
                if args.use_tta:
                    waveforms_modified = apply_tta(config, model, mix_modified, waveforms_orig, device, args.model_type)
                waveforms_orig['instrumental_phaseremix'] = mix_orig + mix_modified_ - waveforms_modified[instr]

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

def proc_folder(args, use_tensorrt=False):
    """
    Process folder with optional TensorRT backend.
    
    Parameters:
    ----------
    args : list or None
        Command line arguments
    use_tensorrt : bool
        Use TensorRT backend if available
    """
    parser = argparse.ArgumentParser(description=i18n("proc_folder_description"))
    parser.add_argument("--model_type", type=str, default='mdx23c', help=i18n("model_type_help"))
    parser.add_argument("--config_path", type=str, help=i18n("config_path_help"))
    parser.add_argument("--demud_phaseremix_inst", action='store_true', help=i18n("demud_phaseremix_help"))
    parser.add_argument("--start_check_point", type=str, default='', help=i18n("start_checkpoint_help"))
    parser.add_argument("--input_folder", type=str, help=i18n("input_folder_help"))
    parser.add_argument("--audio_path", type=str, help=i18n("audio_path_help"))
    parser.add_argument("--store_dir", type=str, default="", help=i18n("store_dir_help"))
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help=i18n("device_ids_help"))
    parser.add_argument("--extract_instrumental", action='store_true', help=i18n("extract_instrumental_help"))
    parser.add_argument("--disable_detailed_pbar", action='store_true', help=i18n("disable_detailed_pbar_help"))
    parser.add_argument("--force_cpu", action='store_true', help=i18n("force_cpu_help"))
    parser.add_argument("--flac_file", action='store_true', help=i18n("flac_file_help"))
    parser.add_argument("--export_format", type=str, choices=['wav FLOAT', 'flac PCM_16', 'flac PCM_24'], default='flac PCM_24', help=i18n("export_format_help"))
    parser.add_argument("--pcm_type", type=str, choices=['PCM_16', 'PCM_24'], default='PCM_24', help=i18n("pcm_type_help"))
    parser.add_argument("--use_tta", action='store_true', help=i18n("use_tta_help"))
    parser.add_argument("--lora_checkpoint", type=str, default='', help=i18n("lora_checkpoint_help"))
    parser.add_argument("--chunk_size", type=int, default=1000000, help="Inference chunk size")
    parser.add_argument("--overlap", type=int, default=4, help="Inference overlap factor")
    parser.add_argument("--use_pytorch_optimized", action='store_true', help="Use optimized PyTorch backend")
    parser.add_argument("--optimize_mode", type=str, choices=['default', 'compile', 'jit', 'channels_last'], default='default', help="PyTorch optimization mode")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    device = "cpu"
    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        print(i18n("cuda_available"))
        device = f'cuda:{args.device_ids[0]}' if type(args.device_ids) == list else f'cuda:{args.device_ids}'
    elif torch.backends.mps.is_available():
         device = "mps"

    print(i18n("using_device").format(device))

    model_load_start_time = time.time()
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(args.model_type, args.config_path)

    if args.start_check_point != '':
        load_start_checkpoint(args, model, type_='inference')

    print(i18n("instruments_print").format(config.training.instruments))

    if type(args.device_ids) == list and len(args.device_ids) > 1 and not args.force_cpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    model = model.to(device)

    print(i18n("model_load_time").format(time.time() - model_load_start_time))

    # Check which backend to use
    use_pytorch_opt = args.use_pytorch_optimized
    
    if use_pytorch_opt and PYTORCH_OPTIMIZED_AVAILABLE:
        print(f"\n🔥 Using optimized PyTorch backend (mode: {args.optimize_mode})")
        print("   To use standard inference, remove --use_pytorch_optimized flag")
        from inference_pytorch import proc_folder_pytorch_optimized
        # Recreate args for optimized PyTorch inference
        sys.argv = sys.argv[:1]  # Keep only script name
        for key, value in vars(args).items():
            if value is not None and value is not False:
                if isinstance(value, bool):
                    sys.argv.append(f"--{key}")
                elif isinstance(value, list):
                    sys.argv.append(f"--{key}")
                    sys.argv.extend(map(str, value))
                else:
                    sys.argv.extend([f"--{key}", str(value)])
        proc_folder_pytorch_optimized(None)
    else:
        run_folder(model, args, config, device, verbose=False)

if __name__ == "__main__":
    proc_folder(None)
