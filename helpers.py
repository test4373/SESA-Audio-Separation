import os
import shutil
import glob
import re
import subprocess
import random
import yaml
from pathlib import Path
import torch
import gradio as gr
import threading
import time
import librosa
import soundfile as sf
import numpy as np
import requests
import json
import locale
from datetime import datetime
import yt_dlp
import validators
from pytube import YouTube

# Google API imports (optional - for Colab/Google Drive support)
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from google.oauth2.credentials import Credentials
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    build = None
    MediaIoBaseDownload = None
    Credentials = None

import io
import math
import hashlib
import gc
import psutil
import concurrent.futures
from tqdm import tqdm
import tempfile
from urllib.parse import urlparse, quote
import argparse
from tqdm.auto import tqdm
import torch.nn as nn
from model import get_model_config, MODEL_CONFIGS, get_all_model_configs_with_custom, load_custom_models
from assets.i18n.i18n import I18nAuto
import matchering as mg
from scipy.signal import find_peaks

i18n = I18nAuto()

# Temel dizinler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OLD_OUTPUT_DIR = os.path.join(BASE_DIR, "old_output")
AUTO_ENSEMBLE_TEMP = os.path.join(BASE_DIR, "auto_ensemble_temp")
AUTO_ENSEMBLE_OUTPUT = os.path.join(BASE_DIR, "ensemble_folder")
VIDEO_TEMP = os.path.join(BASE_DIR, "video_temp")
ENSEMBLE_DIR = os.path.join(BASE_DIR, "ensemble")
COOKIE_PATH = os.path.join(BASE_DIR, "cookies.txt")
INFERENCE_SCRIPT_PATH = os.path.join(BASE_DIR, "inference.py")

def extract_model_name_from_checkpoint(checkpoint_path):
    if not checkpoint_path:
        return "Unknown"
    base_name = os.path.basename(checkpoint_path)
    model_name = os.path.splitext(base_name)[0]
    return model_name.strip()

for directory in [BASE_DIR, INPUT_DIR, OUTPUT_DIR, OLD_OUTPUT_DIR, AUTO_ENSEMBLE_TEMP, AUTO_ENSEMBLE_OUTPUT, VIDEO_TEMP, ENSEMBLE_DIR]:
    os.makedirs(directory, exist_ok=True)

class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)

def tuple_constructor(loader, node):
    """YAML'dan bir tuple yükler."""
    values = loader.construct_sequence(node)
    return tuple(values)

yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)

def clean_model(model):
    """
    Cleans a model name by removing unwanted characters like ⭐ and extra whitespace.
    
    Args:
        model (str): The model name to clean.
    
    Returns:
        str: The cleaned model name, or None if input is invalid.
    """
    if not model or not isinstance(model, str):
        return None
    # Remove ⭐ and extra whitespace
    cleaned = model.replace("⭐", "").strip()
    # Remove any additional unwanted characters if needed
    cleaned = cleaned.replace("\t", " ").replace("\n", " ")
    return cleaned

def get_original_category(translated_category):
    all_configs = get_all_model_configs_with_custom()
    for original_cat in all_configs.keys():
        if i18n(original_cat) == translated_category:
            return original_cat
    return None

def clamp_percentage(value):
    """Clamp percentage values to the 0-100 range."""
    try:
        return min(max(float(value), 0), 100)
    except (ValueError, TypeError):
        print(f"Warning: Invalid percentage value {value}, defaulting to 0")
        return 0    

def update_model_dropdown(category, favorites=None):
    # Get all configs including custom models
    all_configs = get_all_model_configs_with_custom()
    # Map translated category back to English
    eng_cat = next((k for k in all_configs.keys() if i18n(k) == category), list(all_configs.keys())[0])
    models = all_configs.get(eng_cat, {})
    choices = []
    favorite_models = []
    non_favorite_models = []
    
    for model in models:
        model_name = f"{model} ⭐" if favorites and model in favorites else model
        if favorites and model in favorites:
            favorite_models.append(model_name)
        else:
            non_favorite_models.append(model_name)
    
    choices = favorite_models + non_favorite_models
    return {"choices": choices}

def get_model_categories():
    """Get all model categories including Custom Models if any exist."""
    all_configs = get_all_model_configs_with_custom()
    return list(all_configs.keys())

def handle_file_upload(uploaded_file, file_path, is_auto_ensemble=False):
    clear_temp_folder("/tmp", exclude_items=["gradio", "config.json"])
    clear_directory(INPUT_DIR)
    os.makedirs(INPUT_DIR, exist_ok=True)
    clear_directory(INPUT_DIR)
    if uploaded_file:
        target_path = save_uploaded_file(uploaded_file, is_input=True)
        return target_path, target_path
    elif file_path and os.path.exists(file_path):
        target_path = os.path.join(INPUT_DIR, os.path.basename(file_path))
        shutil.copy(file_path, target_path)
        return target_path, target_path
    return None, None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def clear_directory(directory):
    """Verilen dizindeki tüm dosyaları siler."""
    files = glob.glob(os.path.join(directory, '*'))
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(i18n("file_deletion_error").format(f, e))

def clear_temp_folder(folder_path, exclude_items=None):
    """Dizinin içeriğini güvenli bir şekilde temizler ve belirtilen öğeleri korur."""
    try:
        if not os.path.exists(folder_path):
            print(i18n("directory_not_exist_warning").format(folder_path))
            return False
        if not os.path.isdir(folder_path):
            print(i18n("not_a_directory_warning").format(folder_path))
            return False
        exclude_items = exclude_items or []
        for item_name in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item_name)
            if item_name in exclude_items:
                continue
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(i18n("item_deletion_error").format(item_path, e))
        return True
    except Exception as e:
        print(i18n("critical_error").format(e))
        return False

def clear_old_output():
    old_output_folder = os.path.join(BASE_DIR, 'old_output')
    try:
        if not os.path.exists(old_output_folder):
            return i18n("old_output_not_exist")
        shutil.rmtree(old_output_folder)
        os.makedirs(old_output_folder, exist_ok=True)
        return i18n("old_outputs_cleared")
    except Exception as e:
        return i18n("error").format(e)

def shorten_filename(filename, max_length=30):
    """Dosya adını belirtilen maksimum uzunluğa kısaltır."""
    base, ext = os.path.splitext(filename)
    if len(base) <= max_length:
        return filename
    return base[:15] + "..." + base[-10:] + ext

def clean_filename(title):
    """Dosya adından özel karakterleri kaldırır."""
    return re.sub(r'[^\w\-_\. ]', '', title).strip()

def sanitize_filename(filename):
    base, ext = os.path.splitext(filename)
    base = re.sub(r'\.+', '_', base)
    base = re.sub(r'[#<>:"/\\|?*]', '_', base)
    base = re.sub(r'\s+', '_', base)
    base = re.sub(r'_+', '_', base)
    base = base.strip('_')
    return f"{base}{ext}"

def convert_to_wav(file_path):
    """Ses dosyasını WAV formatına dönüştürür."""
    original_filename = os.path.basename(file_path)
    filename, ext = os.path.splitext(original_filename)
    if ext.lower() == '.wav':
        return file_path
    wav_output = os.path.join(ENSEMBLE_DIR, f"{filename}.wav")
    try:
        command = [
            'ffmpeg', '-y', '-i', file_path,
            '-acodec', 'pcm_s16le', '-ar', '44100', wav_output
        ]
        subprocess.run(command, check=True, capture_output=True)
        return wav_output
    except subprocess.CalledProcessError as e:
        print(i18n("ffmpeg_error").format(e.returncode, e.stderr.decode()))
        return None

def generate_random_port():
    """Rastgele bir port numarası oluşturur."""
    return random.randint(1000, 9000)

def save_segment(audio, sr, path):
    """
    Save audio segment to a file.
    
    Args:
        audio (np.ndarray): Audio data.
        sr (int): Sample rate.
        path (str): Output file path.
    """
    sf.write(path, audio, sr)

def run_matchering(reference_path, target_path, output_path, passes=1, bit_depth=24):
    """
    Run Matchering to master the target audio using the reference audio.
    
    Args:
        reference_path (str): Path to the reference audio (clear segment).
        target_path (str): Path to the target audio to be mastered.
        output_path (str): Path for the mastered output.
        passes (int): Number of Matchering passes (1-4).
        bit_depth (int): Output bit depth (16 or 24).
    
    Returns:
        str: Path to the mastered output file.
    """
    # Ensure inputs are WAV files
    ref_audio, sr = librosa.load(reference_path, sr=44100, mono=False)
    tgt_audio, sr = librosa.load(target_path, sr=44100, mono=False)
    
    # Save temporary WAV files
    temp_ref = os.path.join(tempfile.gettempdir(), "matchering_ref.wav")
    temp_tgt = os.path.join(tempfile.gettempdir(), "matchering_tgt.wav")
    save_segment(ref_audio.T if ref_audio.ndim > 1 else ref_audio, sr, temp_ref)
    save_segment(tgt_audio.T if tgt_audio.ndim > 1 else tgt_audio, sr, temp_tgt)
    
    # Configure Matchering with default settings
    config = mg.Config()  # No parameters, use defaults
    
    # Select bit depth for output
    result_format = mg.pcm24 if bit_depth == 24 else mg.pcm16
    
    # Run Matchering for multiple passes
    current_tgt = temp_tgt
    for i in range(passes):
        temp_out = os.path.join(tempfile.gettempdir(), f"matchering_out_pass_{i}.wav")
        mg.process(
            reference=temp_ref,
            target=current_tgt,
            results=[result_format(temp_out)],  # Bit depth control
            config=config
        )
        current_tgt = temp_out
    
    # Move final output to desired path
    shutil.move(current_tgt, output_path)
    
    # Clean up temporary files
    for temp_file in [temp_ref, temp_tgt] + [os.path.join(tempfile.gettempdir(), f"matchering_out_pass_{i}.wav") for i in range(passes-1)]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    return output_path     

def find_clear_segment(audio_path, segment_duration=15, sr=44100):
    """
    Find the clearest (high-energy, low-noise) segment in an audio file.
    
    Args:
        audio_path (str): Path to the original audio file.
        segment_duration (float): Duration of the segment to extract (in seconds).
        sr (int): Sample rate for loading audio.
    
    Returns:
        tuple: (start_time, end_time, segment_audio) of the clearest segment.
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=sr, mono=True)
    
    # Compute RMS energy in windows
    window_size = int(5 * sr)  # 5-second windows
    hop_length = window_size // 2
    rms = librosa.feature.rms(y=audio, frame_length=window_size, hop_length=hop_length)[0]
    
    # Compute spectral flatness for noise detection
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=window_size, hop_length=hop_length)[0]
    
    # Combine metrics: high RMS and low flatness indicate clear, high-energy segments
    score = rms / (flatness + 1e-6)  # Avoid division by zero
    
    # Find peaks in the score
    peaks, _ = find_peaks(score, height=np.mean(score), distance=5)
    if len(peaks) == 0:
        # Fallback: Use the middle of the track
        peak_idx = len(score) // 2
    else:
        peak_idx = peaks[np.argmax(score[peaks])]
    
    # Calculate start and end times
    start_sample = peak_idx * hop_length
    end_sample = start_sample + int(segment_duration * sr)
    
    # Ensure the segment fits within the audio
    if end_sample > len(audio):
        end_sample = len(audio)
        start_sample = max(0, end_sample - int(segment_duration * sr))
    
    start_time = start_sample / sr
    end_time = end_sample / sr
    segment_audio = audio[start_sample:end_sample]
    
    return start_time, end_time, segment_audio

def update_file_list():
    output_files = glob.glob(os.path.join(OUTPUT_DIR, "*.wav"))
    old_output_files = glob.glob(os.path.join(OLD_OUTPUT_DIR, "*.wav"))
    files = output_files + old_output_files
    return gr.Dropdown(choices=files)

def save_uploaded_file(uploaded_file, is_input=False, target_dir=None):
    """Yüklenen dosyayı belirtilen dizine kaydeder."""
    media_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.mp4']
    target_dir = target_dir or (INPUT_DIR if is_input else OUTPUT_DIR)
    timestamp_patterns = [
        r'_\d{8}_\d{6}_\d{6}$', r'_\d{14}$', r'_\d{10}$', r'_\d+$'
    ]

    if hasattr(uploaded_file, 'name'):
        original_filename = os.path.basename(uploaded_file.name)
    else:
        original_filename = os.path.basename(str(uploaded_file))

    if is_input:
        base_filename = original_filename
        for pattern in timestamp_patterns:
            base_filename = re.sub(pattern, '', base_filename)
        for ext in media_extensions:
            base_filename = base_filename.replace(ext, '')
        file_ext = next(
            (ext for ext in media_extensions if original_filename.lower().endswith(ext)),
            '.wav'
        )
        clean_filename = f"{base_filename.strip('_- ')}{file_ext}"
    else:
        clean_filename = original_filename

    target_path = os.path.join(target_dir, clean_filename)
    os.makedirs(target_dir, exist_ok=True)

    if os.path.exists(target_path):
        os.remove(target_path)

    if hasattr(uploaded_file, 'read'):
        with open(target_path, "wb") as f:
            f.write(uploaded_file.read())
    else:
        shutil.copy(uploaded_file, target_path)

    print(i18n("file_saved_successfully").format(os.path.basename(target_path)))
    return target_path

def move_old_files(output_folder):
    """Eski dosyaları old_output dizinine taşır."""
    os.makedirs(OLD_OUTPUT_DIR, exist_ok=True)
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        if os.path.isfile(file_path):
            new_filename = f"{os.path.splitext(filename)[0]}_old{os.path.splitext(filename)[1]}"
            new_file_path = os.path.join(OLD_OUTPUT_DIR, new_filename)
            shutil.move(file_path, new_file_path)
