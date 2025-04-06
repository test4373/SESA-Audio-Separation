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
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import math
import hashlib
import gc
import psutil
import concurrent.futures
from tqdm import tqdm
from google.oauth2.credentials import Credentials
import tempfile
from urllib.parse import urlparse, quote
import argparse
from tqdm.auto import tqdm
import torch.nn as nn
from model import get_model_config, MODEL_CONFIGS
from assets.i18n.i18n import I18nAuto

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

def get_original_category(translated_category):
    for original_cat in MODEL_CONFIGS.keys():
        if i18n(original_cat) == translated_category:
            return original_cat
    return None

def update_model_dropdown(category):
    original_category = get_original_category(category)
    if original_category is None:
        raise ValueError(f"Kategori bulunamadı: {category}")
    return gr.Dropdown(choices=list(MODEL_CONFIGS[original_category].keys()), label=i18n("model"))

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
