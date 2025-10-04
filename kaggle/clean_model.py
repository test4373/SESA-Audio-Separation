# Kaggle için gerekli kütüphaneleri yükleme
import os
import glob
import subprocess
import time
import gc
import shutil
import sys
from datetime import datetime
import torch
import yaml
import numpy as np
import requests
import json
import locale
import re
import psutil
import concurrent.futures
from tqdm import tqdm
from urllib.parse import urlparse, quote

# Ses işleme kütüphaneleri
!pip install librosa soundfile
import librosa
import soundfile as sf

# Uyarıları görmezden gel
import warnings
warnings.filterwarnings("ignore")

# Kaggle dizin ayarları
BASE_DIR = "/kaggle/working/"  # Kaggle'da çalışma dizini
INFERENCE_PATH = os.path.join(BASE_DIR, "inference.py")  # inference.py varsa buraya kopyalanmalı
OUTPUT_DIR = os.path.join(BASE_DIR, "output")  # Çıkış dizini
AUTO_ENSEMBLE_OUTPUT = os.path.join(BASE_DIR, "ensemble_output")  # Ensemble çıkış dizini

# Çıkış dizinlerini oluştur
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUTO_ENSEMBLE_OUTPUT, exist_ok=True)

def clean_model_name(model):
    """
    Clean and standardize model names for filename
    """
    model_name_mapping = {
        'VOCALS-InstVocHQ': 'InstVocHQ',
        'VOCALS-MelBand-Roformer (by KimberleyJSN)': 'KimberleyJSN',
        'VOCALS-BS-Roformer_1297 (by viperx)': 'VOCALS_BS_Roformer1297',
        'VOCALS-BS-Roformer_1296 (by viperx)': 'VOCALS-BS-Roformer_1296',
        'VOCALS-BS-RoformerLargev1 (by unwa)': 'UnwaLargeV1',
        'VOCALS-Mel-Roformer big beta 4 (by unwa)': 'UnwaBigBeta4',
        'VOCALS-Melband-Roformer BigBeta5e (by unwa)': 'UnwaBigBeta5e',
        'INST-Mel-Roformer v1 (by unwa)': 'UnwaInstV1',
        'INST-Mel-Roformer v2 (by unwa)': 'UnwaInstV2',
        'INST-VOC-Mel-Roformer a.k.a. duality (by unwa)': 'UnwaDualityV1',
        'INST-VOC-Mel-Roformer a.k.a. duality v2 (by unwa)': 'UnwaDualityV2',
        'KARAOKE-MelBand-Roformer (by aufr33 & viperx)': 'KaraokeMelBandRoformer',
        'VOCALS-VitLarge23 (by ZFTurbo)': 'VitLarge23',
        'VOCALS-MelBand-Roformer (by Becruily)': 'BecruilyVocals',
        'INST-MelBand-Roformer (by Becruily)': 'BecruilyInst',
        'VOCALS-MelBand-Roformer Kim FT (by Unwa)': 'KimFT',
        'INST-MelBand-Roformer Kim FT (by Unwa)': 'KimFTInst',
        'OTHER-BS-Roformer_1053 (by viperx)': 'OtherViperx1053',
        'CROWD-REMOVAL-MelBand-Roformer (by aufr33)': 'CrowdRemovalRoformer',
        'CINEMATIC-BandIt_Plus (by kwatcharasupat)': 'CinematicBandItPlus',
        'DRUMSEP-MDX23C_DrumSep_6stem (by aufr33 & jarredou)': 'DrumSepMDX23C',
        '4STEMS-SCNet_MUSDB18 (by starrytong)': 'FourStemsSCNet',
        'DE-REVERB-MDX23C (by aufr33 & jarredou)': 'DeReverbMDX23C',
        'DENOISE-MelBand-Roformer-1 (by aufr33)': 'DenoiseMelBand1',
        'DENOISE-MelBand-Roformer-2 (by aufr33)': 'DenoiseMelBand2',
        'INST-MelBand-Roformer (by Becruily)': 'BecruilyInst',
        '4STEMS-SCNet_XL_MUSDB18 (by ZFTurbo)': 'FourStemsSCNetXL',
        '4STEMS-SCNet_Large (by starrytong)': 'FourStemsSCNetLarge',
        '4STEMS-BS-Roformer_MUSDB18 (by ZFTurbo)': 'FourStemsBSRoformer',
        'DE-REVERB-MelBand-Roformer aggr./v2/19.1729 (by anvuew)': 'DeReverbMelBandAggr',
        'DE-REVERB-Echo-MelBand-Roformer (by Sucial)': 'DeReverbEchoMelBand',
        'bleed_suppressor_v1 (by unwa)': 'BleedSuppressorV1',
        'inst_v1e (by unwa)': 'InstV1E',
        'inst_gabox (by Gabox)': 'InstGabox',
        'inst_gaboxBV1 (by Gabox)': 'InstGaboxBV1',
        'inst_gaboxBV2 (by Gabox)': 'InstGaboxBV2',
        'inst_gaboxBFV1 (by Gabox)': 'InstGaboxBFV1',
        'inst_gaboxFV2 (by Gabox)': 'InstGaboxFV2',
        'inst_gaboxFV1 (by Gabox)': 'InstGaboxFV1',
        'dereverb_mel_band_roformer_less_aggressive_anvuew': 'DereverbMelBandRoformerLessAggressive',
        'dereverb_mel_band_roformer_anvuew': 'DereverbMelBandRoformer',
        'VOCALS-Male Female-BS-RoFormer Male Female Beta 7_2889 (by aufr33)': 'MaleFemale-BS-RoFormer-(by aufr33)',
        'VOCALS-MelBand-Roformer (by Becruily)': 'Vocals-MelBand-Roformer-(by Becruily)',
        'VOCALS-MelBand-Roformer Kim FT 2 (by Unwa)': 'Vocals-MelBand-Roformer-KİM-FT-2(by Unwa)',
        'voc_gaboxMelRoformer (by Gabox)': 'voc_gaboxMelRoformer',
        'voc_gaboxBSroformer (by Gabox)': 'voc_gaboxBSroformer',
        'voc_gaboxMelRoformerFV1 (by Gabox)': 'voc_gaboxMelRoformerFV1',
        'voc_gaboxMelRoformerFV2 (by Gabox)': 'voc_gaboxMelRoformerFV2',
        'SYH99999/MelBandRoformerSYHFTB1(by Amane)': 'MelBandRoformerSYHFTB1',
        'inst_V5 (by Gabox)': 'INSTV5-(by Gabox)',
        'inst_Fv4Noise (by Gabox)': 'Inst_Fv4Noise-(by Gabox)',
        'Intrumental_Gabox (by Gabox)': 'Intrumental_Gabox-(by Gabox)',
        'inst_GaboxFv3 (by Gabox)': 'INST_GaboxFv3-(by Gabox)',
        'SYH99999/MelBandRoformerSYHFTB1_Model1 (by Amane)': 'MelBandRoformerSYHFTB1_model1',
        'SYH99999/MelBandRoformerSYHFTB1_Model2 (by Amane)': 'MelBandRoformerSYHFTB1_model2',
        'SYH99999/MelBandRoformerSYHFTB1_Model3 (by Amane)': 'MelBandRoformerSYHFTB1_model3',
        'VOCALS-MelBand-Roformer Kim FT 2 Blendless (by unwa)': 'VOCALS-MelBand-Roformer-Kim-FT-2-Blendless-(by unwa)',
        'inst_gaboxFV6 (by Gabox)': 'inst_gaboxFV6-(by Gabox)',
        'denoisedebleed (by Gabox)': 'denoisedebleed-(by Gabox)',
        'INSTV5N (by Gabox)': 'INSTV5N_(by Gabox)',
        'Voc_Fv3 (by Gabox)': 'Voc_Fv3_(by Gabox)',
        'MelBandRoformer4StemFTLarge (SYH99999)': 'MelBandRoformer4StemFTLarge_(SYH99999)',
        'dereverb_mel_band_roformer_mono (by anvuew)': 'dereverb_mel_band_roformer_mono_(by anvuew)',
        'INSTV6N (by Gabox)': 'INSTV6N_(by Gabox)',
        'KaraokeGabox': 'KaraokeGabox',
        'FullnessVocalModel (by Amane)': 'FullnessVocalModel',
        'Inst_GaboxV7 (by Gabox)': 'Inst_GaboxV7_(by Gabox)',
    }

    if model in model_name_mapping:
        return model_name_mapping[model]
    
    cleaned = re.sub(r'\s*\(.*?\)', '', model)  # Remove parenthetical info
    cleaned = cleaned.replace('-', '_')
    cleaned = ''.join(char for char in cleaned if char.isalnum() or char == '_')
    
    return cleaned

def shorten_filename(filename, max_length=30):
    """
    Shortens a filename to a specified maximum length
    """
    base, ext = os.path.splitext(filename)
    if len(base) <= max_length:
        return filename
    shortened = base[:15] + "..." + base[-10:] + ext
    return shortened

def clean_filename(filename):
    """
    Temizlenmiş dosya adını döndürür
    """
    cleanup_patterns = [
        r'_\d{8}_\d{6}_\d{6}$',  # _20231215_123456_123456
        r'_\d{14}$',              # _20231215123456
        r'_\d{10}$',              # _1702658400
        r'_\d+$'                  # Herhangi bir sayı
    ]
    
    base, ext = os.path.splitext(filename)
    for pattern in cleanup_patterns:
        base = re.sub(pattern, '', base)
    
    file_types = ['vocals', 'instrumental', 'drum', 'bass', 'other', 'effects', 'speech', 'music', 'dry', 'male', 'female']
    for type_keyword in file_types:
        base = base.replace(f'_{type_keyword}', '')
    
    detected_type = None
    for type_keyword in file_types:
        if type_keyword in base.lower():
            detected_type = type_keyword
            break
    
    clean_base = base.strip('_- ')
    return clean_base, detected_type, ext

# Örnek kullanım (Kaggle veri setinden dosya okuma)
# Veri setiniz /kaggle/input/ altında olmalı
# Örnek: input_file = "/kaggle/input/your-dataset/sample_audio.wav"
# clean_base, detected_type, ext = clean_filename("example_vocals_20231215.wav")
# print(clean_base, detected_type, ext)
