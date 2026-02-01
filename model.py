import os
import yaml
import json
import re
from urllib.parse import quote
from pathlib import Path

# Temel dizin ve checkpoint dizini sabit olarak tanımlanıyor
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'ckpts')
CUSTOM_MODELS_FILE = os.path.join(BASE_DIR, 'assets', 'custom_models.json')

# Supported model types for auto-detection and manual selection
SUPPORTED_MODEL_TYPES = [
    'bs_roformer',
    'bs_roformer_custom',
    'mel_band_roformer',
    'mdx23c',
    'bandit_v2',
    'scnet',
    'htdemucs',
    'torchseg'
]

def detect_model_type_from_url(checkpoint_url, config_url=None):
    """Auto-detect model type from URL patterns."""
    urls_to_check = [checkpoint_url]
    if config_url:
        urls_to_check.append(config_url)
    
    combined_text = ' '.join(urls_to_check).lower()
    
    patterns = [
        (r'bs[-_]?roformer[-_]?custom|hyperace', 'bs_roformer_custom'),
        (r'bs[-_]?roformer|bsroformer', 'bs_roformer'),
        (r'mel[-_]?band[-_]?roformer|melbandroformer|mbr', 'mel_band_roformer'),
        (r'mdx23c', 'mdx23c'),
        (r'bandit[-_]?v?2?', 'bandit_v2'),
        (r'scnet', 'scnet'),
        (r'htdemucs|demucs', 'htdemucs'),
        (r'torchseg', 'torchseg'),
    ]
    
    for pattern, model_type in patterns:
        if re.search(pattern, combined_text):
            return model_type
    return None

def detect_model_type_from_config(config_url):
    """Try to detect model type by downloading and parsing config YAML."""
    try:
        import requests
        response = requests.get(config_url, timeout=10)
        if response.status_code == 200:
            config_data = yaml.safe_load(response.text)
            if 'model_type' in config_data:
                return config_data['model_type']
            if 'model' in config_data and 'model_type' in config_data['model']:
                return config_data['model']['model_type']
    except Exception as e:
        print(f"Could not detect model type from config: {e}")
    return None

def load_custom_models():
    """Load custom models from JSON file."""
    if not os.path.exists(CUSTOM_MODELS_FILE):
        return {}
    try:
        with open(CUSTOM_MODELS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading custom models: {e}")
        return {}

def save_custom_models(models):
    """Save custom models to JSON file."""
    os.makedirs(os.path.dirname(CUSTOM_MODELS_FILE), exist_ok=True)
    with open(CUSTOM_MODELS_FILE, 'w', encoding='utf-8') as f:
        json.dump(models, f, indent=2, ensure_ascii=False)

def add_custom_model(model_name, model_type, checkpoint_url, config_url, custom_model_url=None, auto_detect=True):
    """Add a new custom model."""
    if not model_name or not model_name.strip():
        return False, "Model name is required"
    if not checkpoint_url or not checkpoint_url.strip():
        return False, "Checkpoint URL is required"
    if not config_url or not config_url.strip():
        return False, "Config URL is required"
    
    model_name = model_name.strip()
    checkpoint_url = checkpoint_url.strip()
    config_url = config_url.strip()
    custom_model_url = custom_model_url.strip() if custom_model_url else None
    
    if auto_detect and (not model_type or model_type == "auto"):
        detected_type = detect_model_type_from_url(checkpoint_url, config_url)
        if not detected_type:
            detected_type = detect_model_type_from_config(config_url)
        if detected_type:
            model_type = detected_type
            print(f"Auto-detected model type: {model_type}")
        else:
            return False, "Could not auto-detect model type. Please select manually."
    
    if model_type not in SUPPORTED_MODEL_TYPES:
        return False, f"Unsupported model type: {model_type}"
    
    checkpoint_filename = os.path.basename(checkpoint_url.split('?')[0])
    config_filename = f"config_{model_name.replace(' ', '_').lower()}.yaml"
    
    models = load_custom_models()
    if model_name in models:
        return False, f"Model '{model_name}' already exists"
    
    models[model_name] = {
        'model_type': model_type,
        'checkpoint_url': checkpoint_url,
        'config_url': config_url,
        'custom_model_url': custom_model_url,
        'checkpoint_filename': checkpoint_filename,
        'config_filename': config_filename,
        'needs_conf_edit': True
    }
    save_custom_models(models)
    return True, f"Model '{model_name}' added successfully"

def delete_custom_model(model_name):
    """Delete a custom model."""
    models = load_custom_models()
    if model_name not in models:
        return False, f"Model '{model_name}' not found"
    
    model_config = models[model_name]
    checkpoint_path = os.path.join(CHECKPOINT_DIR, model_config.get('checkpoint_filename', ''))
    config_path = os.path.join(CHECKPOINT_DIR, model_config.get('config_filename', ''))
    
    try:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        if os.path.exists(config_path):
            os.remove(config_path)
    except Exception as e:
        print(f"Warning: Could not delete model files: {e}")
    
    del models[model_name]
    save_custom_models(models)
    return True, f"Model '{model_name}' deleted successfully"

def get_custom_models_list():
    """Get list of custom model names with their types."""
    models = load_custom_models()
    return [(name, config.get('model_type', 'unknown')) for name, config in models.items()]

def conf_edit(config_path, chunk_size, overlap):
    """Edits the configuration file with chunk size and overlap."""
    full_config_path = os.path.join(CHECKPOINT_DIR, os.path.basename(config_path))
    if not os.path.exists(full_config_path):
        raise FileNotFoundError(f"Configuration file not found: {full_config_path}")
    
    with open(full_config_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    if 'use_amp' not in data.keys():
        data['training']['use_amp'] = True

    data['audio']['chunk_size'] = chunk_size
    data['inference']['num_overlap'] = overlap
    if data['inference']['batch_size'] == 1:
        data['inference']['batch_size'] = 2

    print(f"Using custom overlap and chunk_size: overlap={overlap}, chunk_size={chunk_size}")
    with open(full_config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, Dumper=yaml.Dumper)

def download_file(url, path=None, target_filename=None):
    """Downloads a file from a URL.
    
    Args:
        url: The URL to download from.
        path: The directory to save the file to. Defaults to CHECKPOINT_DIR.
        target_filename: Optional custom filename to save as. If None, uses filename from URL.
    """
    import requests
    encoded_url = quote(url, safe=':/')
    if path is None:
        path = CHECKPOINT_DIR
    os.makedirs(path, exist_ok=True)
    # Use custom target filename if provided, otherwise extract from URL
    filename = target_filename if target_filename else os.path.basename(encoded_url)
    file_path = os.path.join(path, filename)
    if os.path.exists(file_path):
        print(f"File '{filename}' already exists at '{path}'.")
        return
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"File '{filename}' downloaded successfully")
        else:
            print(f"Error downloading '{filename}': Status code {response.status_code}")
    except Exception as e:
        print(f"Error downloading file '{filename}' from '{url}': {e}")

# Model konfigurasyonlarını kategorize bir sözlükte tut
MODEL_CONFIGS = {
    "Vocal Models": {
        # === NEW MODELS (en üstte) ===
        'bs_roformer_voc_hyperacev2 (by unwa)': {
            'model_type': 'bs_roformer_custom',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_hyperacev2_voc.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'bs_roformer_voc_hyperacev2.ckpt'),
            'download_urls': [
                ('https://huggingface.co/pcunwa/BS-Roformer-HyperACE/resolve/main/v2_voc/config.yaml', 'config_hyperacev2_voc.yaml'),
                'https://huggingface.co/pcunwa/BS-Roformer-HyperACE/resolve/main/v2_voc/bs_roformer_voc_hyperacev2.ckpt'
            ],
            'custom_model_url': 'https://huggingface.co/pcunwa/BS-Roformer-HyperACE/resolve/main/v2_voc/bs_roformer.py',
            'needs_conf_edit': True
        },
        'BS-Roformer-Resurrection (by unwa)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'BS-Roformer-Resurrection-Config.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'BS-Roformer-Resurrection.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection-Config.yaml',
                'https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection.ckpt'
            ],
            'needs_conf_edit': True
        },
        'bs_roformer_revive3e (by unwa)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_revive.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'bs_roformer_revive3e.ckpt'),
            'download_urls': [
                ('https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/config.yaml', 'config_revive.yaml'),
                'https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/bs_roformer_revive3e.ckpt'
            ],
            'needs_conf_edit': True
        },
        'bs_roformer_revive2 (by unwa)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_revive.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'bs_roformer_revive2.ckpt'),
            'download_urls': [
                ('https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/config.yaml', 'config_revive.yaml'),
                'https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/bs_roformer_revive2.ckpt'
            ],
            'needs_conf_edit': True
        },
        'bs_roformer_revive (by unwa)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_revive.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'bs_roformer_revive.ckpt'),
            'download_urls': [
                ('https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/config.yaml', 'config_revive.yaml'),
                'https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/bs_roformer_revive.ckpt'
            ],
            'needs_conf_edit': True
        },
        'karaoke_bs_roformer_anvuew (by anvuew)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'karaoke_bs_roformer_anvuew.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'karaoke_bs_roformer_anvuew.ckpt'),
            'download_urls': [
                'https://huggingface.co/anvuew/karaoke_bs_roformer/resolve/main/karaoke_bs_roformer_anvuew.yaml',
                'https://huggingface.co/anvuew/karaoke_bs_roformer/resolve/main/karaoke_bs_roformer_anvuew.ckpt'
            ],
            'needs_conf_edit': True
        },
        # === EXISTING MODELS ===
        'VOCALS-big_beta6X (by Unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'big_beta6x.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'big_beta6x.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6x.yaml',
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6x.ckpt'
            ],
            'needs_conf_edit': False
        },
        'VOCALS-big_beta6 (by Unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'big_beta6.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'big_beta6.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6.yaml',
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta6.ckpt'
            ],
            'needs_conf_edit': False
        },
        'VOCALS-Mel-Roformer FT 3 Preview (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_kimmel_unwa_ft.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'kimmel_unwa_ft3_prev.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml',
                'https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft3_prev.ckpt'
            ],
            'needs_conf_edit': False
        },
        'VOCALS-InstVocHQ': {
            'model_type': 'mdx23c',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_vocals_mdx23c.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model_vocals_mdx23c_sdr_10.17.ckpt'),
            'download_urls': [
                'https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_vocals_mdx23c.yaml',
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt'
            ],
            'needs_conf_edit': False
        },
        'VOCALS-MelBand-Roformer (by KimberleyJSN)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_vocals_mel_band_roformer_kj.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'MelBandRoformer.ckpt'),
            'download_urls': [
                'https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml',
                'https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt'
            ],
            'needs_conf_edit': True
        },
        'VOCALS-BS-Roformer_1297 (by viperx)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'model_bs_roformer_ep_317_sdr_12.9755.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model_bs_roformer_ep_317_sdr_12.9755.ckpt'),
            'download_urls': [
                'https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml',
                'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt'
            ],
            'needs_conf_edit': True
        },
        'VOCALS-BS-Roformer_1296 (by viperx)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'model_bs_roformer_ep_368_sdr_12.9628.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model_bs_roformer_ep_368_sdr_12.9628.ckpt'),
            'download_urls': [
                'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt',
                'https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_368_sdr_12.9628.yaml'
            ],
            'needs_conf_edit': True
        },
        'VOCALS-BS-RoformerLargev1 (by unwa)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_bsrofoL.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'BS-Roformer_LargeV1.ckpt'),
            'download_urls': [
                'https://huggingface.co/jarredou/unwa_bs_roformer/resolve/main/BS-Roformer_LargeV1.ckpt',
                'https://huggingface.co/jarredou/unwa_bs_roformer/raw/main/config_bsrofoL.yaml'
            ],
            'needs_conf_edit': True
        },
        'VOCALS-Mel-Roformer big beta 4 (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_melbandroformer_big_beta4.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'melband_roformer_big_beta4.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta4.ckpt',
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-big/raw/main/config_melbandroformer_big_beta4.yaml'
            ],
            'needs_conf_edit': True
        },
        'VOCALS-Melband-Roformer BigBeta5e (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'big_beta5e.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'big_beta5e.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.ckpt',
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.yaml'
            ],
            'needs_conf_edit': True
        },
        'VOCALS-VitLarge23 (by ZFTurbo)': {
            'model_type': 'segm_models',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_vocals_segm_models.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model_vocals_segm_models_sdr_9.77.ckpt'),
            'download_urls': [
                'https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/refs/heads/main/configs/config_vocals_segm_models.yaml',
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_segm_models_sdr_9.77.ckpt'
            ],
            'needs_conf_edit': False
        },
        'VOCALS-MelBand-Roformer Kim FT (by Unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_kimmel_unwa_ft.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'kimmel_unwa_ft.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft.ckpt',
                'https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml'
            ],
            'needs_conf_edit': True
        },
        'VOCALS-MelBand-Roformer (by Becruily)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_instrumental_becruily.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'mel_band_roformer_vocals_becruily.ckpt'),
            'download_urls': [
                'https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/config_vocals_becruily.yaml',
                'https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/mel_band_roformer_vocals_becruily.ckpt'
            ],
            'needs_conf_edit': True
        },
        'VOCALS-Male Female-BS-RoFormer Male Female Beta 7_2889 (by aufr33)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_chorus_male_female_bs_roformer.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt'),
            'download_urls': [
                'https://huggingface.co/RareSirMix/AIModelRehosting/resolve/main/bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt',
                'https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/config_chorus_male_female_bs_roformer.yaml'
            ],
            'needs_conf_edit': True
        },
        'VOCALS-MelBand-Roformer Kim FT 2 (by Unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_kimmel_unwa_ft.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'kimmel_unwa_ft2.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml',
                'https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft2.ckpt'
            ],
            'needs_conf_edit': True
        },
        'voc_gaboxBSroformer (by Gabox)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'voc_gaboxBSroformer.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'voc_gaboxBSR.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/BSRoformerVocTest/resolve/main/voc_gaboxBSroformer.yaml',
                'https://huggingface.co/GaboxR67/BSRoformerVocTest/resolve/main/voc_gaboxBSR.ckpt'
            ],
            'needs_conf_edit': True
        },
        'voc_gaboxMelReformer (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'voc_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'voc_gabox.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.ckpt'
            ],
            'needs_conf_edit': True
        },
        'voc_gaboxMelReformerFV1 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'voc_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'voc_gaboxFv1.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gaboxFv1.ckpt'
            ],
            'needs_conf_edit': True
        },
        'voc_gaboxMelReformerFV2 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'voc_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'voc_gaboxFv2.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gaboxFv2.ckpt'
            ],
            'needs_conf_edit': True
        },
        'VOCALS-MelBand-Roformer Kim FT 2 Blendless (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_kimmel_unwa_ft.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'kimmel_unwa_ft2_bleedless.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml',
                'https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft2_bleedless.ckpt'
            ],
            'needs_conf_edit': True
        },
        'Voc_Fv3 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'voc_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'voc_Fv3.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_Fv3.ckpt'
            ],
            'needs_conf_edit': True
        },
        'FullnessVocalModel (by Amane)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'FullnessVocalModel.ckpt'),
            'download_urls': [
                'https://huggingface.co/Aname-Tommy/MelBandRoformers/blob/main/config.yaml',
                'https://huggingface.co/Aname-Tommy/MelBandRoformers/blob/main/FullnessVocalModel.ckpt'
            ],
            'needs_conf_edit': True
        },
        'voc_fv4 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'voc_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'voc_fv4.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_fv4.ckpt'
            ],
            'needs_conf_edit': True
        },
        'voc_fv5 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'voc_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'voc_fv5.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_fv5.ckpt'
            ],
            'needs_conf_edit': True
        },
        'voc_fv6 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'voc_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'voc_fv6.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_fv6.ckpt'
            ],
            'needs_conf_edit': True
        },
        'voc_fv7 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'v7.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'voc_fv7.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/v7.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_fv7.ckpt'
            ],
            'needs_conf_edit': True
        },
        'vocfv7beta1 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'voc_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'vocfv7beta1.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/experimental/vocfv7beta1.ckpt'
            ],
            'needs_conf_edit': True
        },
        'vocfv7beta2 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'voc_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'vocfv7beta2.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/experimental/vocfv7beta2.ckpt'
            ],
            'needs_conf_edit': True
        },
        'vocfv7beta3 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'voc_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'vocfv7beta3.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/experimental/vocfv7beta3.ckpt'
            ],
            'needs_conf_edit': True
        },
        'MelBandRoformerSYHFTV3Epsilon (by SYH99999)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_vocals_mel_band_roformer_ft.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'MelBandRoformerSYHFTV3Epsilon.ckpt'),
            'download_urls': [
                'https://huggingface.co/SYH99999/MelBandRoformerSYHFT/resolve/main/config_vocals_mel_band_roformer_ft.yaml',
                'https://huggingface.co/SYH99999/MelBandRoformerSYHFTV3Epsilon/resolve/main/MelBandRoformerSYHFTV3Epsilon.ckpt'
            ],
            'needs_conf_edit': True
        },
        'MelBandRoformerBigSYHFTV1 (by SYH99999)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_big_syhft.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'MelBandRoformerBigSYHFTV1.ckpt'),
            'download_urls': [
                'https://huggingface.co/SYH99999/MelBandRoformerBigSYHFTV1Fast/resolve/main/config.yaml',
                'https://huggingface.co/SYH99999/MelBandRoformerBigSYHFTV1Fast/resolve/main/MelBandRoformerBigSYHFTV1.ckpt'
            ],
            'needs_conf_edit': True
        },
        'model_chorus_bs_roformer_ep_146 (by Sucial)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_chorus_male_female_bs_roformer.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model_chorus_bs_roformer_ep_146_sdr_23.8613.ckpt'),
            'download_urls': [
                'https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/config_chorus_male_female_bs_roformer.yaml',
                'https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/model_chorus_bs_roformer_ep_146_sdr_23.8613.ckpt'
            ],
            'needs_conf_edit': True
        },
        'model_chorus_bs_roformer_ep_267 (by Sucial)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_chorus_male_female_bs_roformer.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt'),
            'download_urls': [
                'https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/config_chorus_male_female_bs_roformer.yaml',
                'https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt'
            ],
            'needs_conf_edit': True
        },
        'BS-Rofo-SW-Fixed (by jarredou)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'BS-Rofo-SW-Fixed.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'BS-Rofo-SW-Fixed.ckpt'),
            'download_urls': [
                'https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.yaml',
                'https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.ckpt'
            ],
            'needs_conf_edit': True
        },
        'BS_ResurrectioN (by Gabox)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'BS-Roformer-Resurrection-Inst-Config.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'BS_ResurrectioN.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection-Inst-Config.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/experimental/BS_ResurrectioN.ckpt'
            ],
            'needs_conf_edit': True
        }
    },
    "Instrumental Models": {
        # === NEW MODELS (en üstte) ===
        'Neo_InstVFX (by natanworkspace)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_neo_inst.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'Neo_InstVFX.ckpt'),
            'download_urls': [
                'https://huggingface.co/natanworkspace/melband_roformer/resolve/main/config_neo_inst.yaml',
                'https://huggingface.co/natanworkspace/melband_roformer/resolve/main/Neo_InstVFX.ckpt'
            ],
            'needs_conf_edit': True
        },
        'BS-Roformer-Resurrection-Inst (by unwa)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'BS-Roformer-Resurrection-Inst-Config.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'BS-Roformer-Resurrection-Inst.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection-Inst-Config.yaml',
                'https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection-Inst.ckpt'
            ],
            'needs_conf_edit': True
        },
        'bs_roformer_inst_hyperacev2 (by unwa)': {
            'model_type': 'bs_roformer_custom',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_hyperacev2_inst.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'bs_roformer_inst_hyperacev2.ckpt'),
            'download_urls': [
                ('https://huggingface.co/pcunwa/BS-Roformer-HyperACE/resolve/main/v2_inst/config.yaml', 'config_hyperacev2_inst.yaml'),
                'https://huggingface.co/pcunwa/BS-Roformer-HyperACE/resolve/main/v2_inst/bs_roformer_inst_hyperacev2.ckpt'
            ],
            'custom_model_url': 'https://huggingface.co/pcunwa/BS-Roformer-HyperACE/resolve/main/v2_inst/bs_roformer.py',
            'needs_conf_edit': True
        },
        'BS-Roformer-Large-Inst (by unwa)': {
            'model_type': 'bs_roformer_custom',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_bs_large_inst.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'bs_large_v2_inst.ckpt'),
            'download_urls': [
                ('https://huggingface.co/pcunwa/BS-Roformer-Large-Inst/resolve/main/config.yaml', 'config_bs_large_inst.yaml'),
                'https://huggingface.co/pcunwa/BS-Roformer-Large-Inst/resolve/main/bs_large_v2_inst.ckpt'
            ],
            'custom_model_url': 'https://huggingface.co/pcunwa/BS-Roformer-Large-Inst/resolve/main/bs_roformer.py',
            'needs_conf_edit': True
        },
        'bs_roformer_fno (by unwa)': {
            'model_type': 'bs_roformer_custom',
            'config_path': os.path.join(CHECKPOINT_DIR, 'bsrofo_fno.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'bs_roformer_fno.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/BS-Roformer-Inst-FNO/resolve/main/bsrofo_fno.yaml',
                'https://huggingface.co/pcunwa/BS-Roformer-Inst-FNO/resolve/main/bs_roformer_fno.ckpt'
            ],
            'custom_model_url': 'https://huggingface.co/listra92/MyModels/resolve/main/misc/bs_roformer.py',
            'needs_conf_edit': True
        },
        'Rifforge_final_sdr_14.24 (by meskvlla33)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_rifforge_full_mesk.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'rifforge_full_sdr_14.2436.ckpt'),
            'download_urls': [
                'https://huggingface.co/meskvlla33/rifforge/resolve/main/config_rifforge_full_mesk.yaml',
                'https://huggingface.co/meskvlla33/rifforge/resolve/main/rifforge_full_sdr_14.2436.ckpt'
            ],
            'needs_conf_edit': True
        },
        # === EXISTING MODELS ===
        'Inst_GaboxFv8 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'Inst_GaboxFv8.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxFv8.ckpt',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml'
            ],
            'needs_conf_edit': True
        },   
        'INST-Mel-Roformer v1 (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_melbandroformer_inst.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'melband_roformer_inst_v1.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v1.ckpt',
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst.yaml'
            ],
            'needs_conf_edit': True
        },
        'INST-Mel-Roformer v1e+ (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_melbandroformer_inst.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'inst_v1e_plus.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e_plus.ckpt',
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst.yaml'
            ],
            'needs_conf_edit': True
        },
        'INST-Mel-Roformer v1+ (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_melbandroformer_inst.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'inst_v1_plus_test.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1_plus_test.ckpt',
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst.yaml'
            ],
            'needs_conf_edit': True
        },
        'INST-Mel-Roformer v2 (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_melbandroformer_inst_v2.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'melband_roformer_inst_v2.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v2.ckpt',
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst_v2.yaml'
            ],
            'needs_conf_edit': True
        },
        'INST-VOC-Mel-Roformer a.k.a. duality (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_melbandroformer_instvoc_duality.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'melband_roformer_instvoc_duality_v1.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvoc_duality_v1.ckpt',
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/raw/main/config_melbandroformer_instvoc_duality.yaml'
            ],
            'needs_conf_edit': True
        },
        'INST-VOC-Mel-Roformer a.k.a. duality v2 (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_melbandroformer_instvoc_duality.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'melband_roformer_instvox_duality_v2.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvox_duality_v2.ckpt',
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/raw/main/config_melbandroformer_instvoc_duality.yaml'
            ],
            'needs_conf_edit': True
        },
        'INST-MelBand-Roformer (by Becruily)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_instrumental_becruily.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'mel_band_roformer_instrumental_becruily.ckpt'),
            'download_urls': [
                'https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/config_instrumental_becruily.yaml',
                'https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt'
            ],
            'needs_conf_edit': True
        },
        'inst_v1e (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_melbandroformer_inst.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'inst_v1e.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e.ckpt',
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml'
            ],
            'needs_conf_edit': True
        },
        'inst_gabox (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'inst_gabox.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.ckpt'
            ],
            'needs_conf_edit': True
        },
        'inst_gaboxBV1 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'inst_gaboxBv1.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxBv1.ckpt'
            ],
            'needs_conf_edit': True
        },
        'inst_gaboxBV2 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'inst_gaboxBv2.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxBv2.ckpt'
            ],
            'needs_conf_edit': True
        },
        'inst_gaboxBFV1 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'gaboxFv1.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv1.ckpt'
            ],
            'needs_conf_edit': True
        },
        'inst_gaboxFV2 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'inst_gaboxFv2.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv2.ckpt'
            ],
            'needs_conf_edit': True
        },
        'inst_Fv3 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'inst_gaboxFv3.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv3.ckpt'
            ],
            'needs_conf_edit': True
        },
        'Intrumental_Gabox (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'intrumental_gabox.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/intrumental_gabox.ckpt'
            ],
            'needs_conf_edit': True
        },
        'inst_Fv4Noise (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'inst_Fv4Noise.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_Fv4Noise.ckpt'
            ],
            'needs_conf_edit': True
        },
        'INSTV5 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'INSTV5.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV5.ckpt'
            ],
            'needs_conf_edit': True
        },
        'inst_gaboxFV1 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'inst_gaboxFv1.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv1.ckpt'
            ],
            'needs_conf_edit': True
        },
        'inst_gaboxFV6 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'INSTV6.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV6.ckpt'
            ],
            'needs_conf_edit': True
        },
        'INSTV5N (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'INSTV5N.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV5N.ckpt'
            ],
            'needs_conf_edit': True
        },
        'INSTV6N (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'INSTV6N.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV6N.ckpt'
            ],
            'needs_conf_edit': True
        },
        'Inst_GaboxV7 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'Inst_GaboxV7.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxV7.ckpt'
            ],
            'needs_conf_edit': True
        },
        'inst_Fv4 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'inst_Fv4.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_Fv4.ckpt'
            ],
            'needs_conf_edit': True
        },
        'INSTV7N (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'INSTV7N.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV7N.ckpt'
            ],
            'needs_conf_edit': True
        },
        'inst_fv7b (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'inst_fv7b.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/experimental/inst_fv7b.ckpt'
            ],
            'needs_conf_edit': True
        },
        'inst_fv7z (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'Inst_GaboxFv7z.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxFv7z.ckpt'
            ],
            'needs_conf_edit': True
        },
        'Inst_GaboxFv9 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'Inst_GaboxFv9.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxFv9.ckpt'
            ],
            'needs_conf_edit': True
        },
        'inst_gaboxFlowersV10 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'v10.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'inst_gaboxFlowersV10.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/v10.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFlowersV10.ckpt'
            ],
            'needs_conf_edit': True
        },
        'Inst_FV8b (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'Inst_FV8b.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/experimental/Inst_FV8b.ckpt'
            ],
            'needs_conf_edit': True
        },
        'Inst_Fv8 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'Inst_Fv8.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/experimental/Inst_Fv8.ckpt'
            ],
            'needs_conf_edit': True
        }
    },
    "4-Stem Models": {
        '4STEMS-SCNet_MUSDB18 (by starrytong)': {
            'model_type': 'scnet',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_musdb18_scnet.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'scnet_checkpoint_musdb18.ckpt'),
            'download_urls': [
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/config_musdb18_scnet.yaml',
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/scnet_checkpoint_musdb18.ckpt'
            ],
            'needs_conf_edit': False
        },
        '4STEMS-SCNet_XL_MUSDB18 (by ZFTurbo)': {
            'model_type': 'scnet',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_musdb18_scnet_xl.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model_scnet_ep_54_sdr_9.8051.ckpt'),
            'download_urls': [
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/config_musdb18_scnet_xl.yaml',
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/model_scnet_ep_54_sdr_9.8051.ckpt'
            ],
            'needs_conf_edit': True
        },
        '4STEMS-SCNet_Large (by starrytong)': {
            'model_type': 'scnet',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_musdb18_scnet_large_starrytong.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'SCNet-large_starrytong_fixed.ckpt'),
            'download_urls': [
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/config_musdb18_scnet_large_starrytong.yaml',
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/SCNet-large_starrytong_fixed.ckpt'
            ],
            'needs_conf_edit': True
        },
        '4STEMS-BS-Roformer_MUSDB18 (by ZFTurbo)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_bs_roformer_384_8_2_485100.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model_bs_roformer_ep_17_sdr_9.6568.ckpt'),
            'download_urls': [
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/config_bs_roformer_384_8_2_485100.yaml',
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt'
            ],
            'needs_conf_edit': True
        },
        'MelBandRoformer4StemFTLarge (SYH99999)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'MelBandRoformer4StemFTLarge.ckpt'),
            'download_urls': [
                'https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/config.yaml',
                'https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/MelBandRoformer4StemFTLarge.ckpt'
            ],
            'needs_conf_edit': True
        }
    },
    "Denoise Models": {
        'DENOISE-MelBand-Roformer-1 (by aufr33)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'model_mel_band_roformer_denoise.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt'),
            'download_urls': [
                'https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt',
                'https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml'
            ],
            'needs_conf_edit': True
        },
        'DENOISE-MelBand-Roformer-2 (by aufr33)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'model_mel_band_roformer_denoise.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt'),
            'download_urls': [
                'https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt',
                'https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml'
            ],
            'needs_conf_edit': True
        },
        'denoisedebleed (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'model_mel_band_roformer_denoise.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'denoisedebleed.ckpt'),
            'download_urls': [
                'https://huggingface.co/poiqazwsx/melband-roformer-denoise/resolve/main/model_mel_band_roformer_denoise.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/denoisedebleed.ckpt'
            ],
            'needs_conf_edit': True
        },    
        'bleed_suppressor_v1 (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_bleed_suppressor_v1.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'bleed_suppressor_v1.ckpt'),
            'download_urls': [
                'https://huggingface.co/ASesYusuf1/MODELS/resolve/main/bleed_suppressor_v1.ckpt',
                'https://huggingface.co/ASesYusuf1/MODELS/resolve/main/config_bleed_suppressor_v1.yaml'
            ],
            'needs_conf_edit': True    
        }
    },
    "Dereverb Models": {
        'DE-REVERB-MDX23C (by aufr33 & jarredou)': {
            'model_type': 'mdx23c',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_dereverb_mdx23c.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'dereverb_mdx23c_sdr_6.9096.ckpt'),
            'download_urls': [
                'https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/dereverb_mdx23c_sdr_6.9096.ckpt',
                'https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/config_dereverb_mdx23c.yaml'
            ],
            'needs_conf_edit': False
        },
        'DE-REVERB-MelBand-Roformer aggr./v2/19.1729 (by anvuew)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'dereverb_mel_band_roformer_anvuew.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt'),
            'download_urls': [
                'https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt',
                'https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml'
            ],
            'needs_conf_edit': True
        },
        'DE-REVERB-Echo-MelBand-Roformer (by Sucial)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_dereverb-echo_mel_band_roformer.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt'),
            'download_urls': [
                'https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt',
                'https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb-echo_mel_band_roformer.yaml'
            ],
            'needs_conf_edit': True
        },
        'dereverb_mel_band_roformer_less_aggressive_anvuew': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'dereverb_mel_band_roformer_anvuew.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt'),
            'download_urls': [
                'https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml',
                'https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt'
            ],
            'needs_conf_edit': True
        },
        'dereverb_mel_band_roformer_anvuew': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'dereverb_mel_band_roformer_anvuew.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt'),
            'download_urls': [
                'https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml',
                'https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt'
            ],
            'needs_conf_edit': True
        },
        'dereverb_mel_band_roformer_mono (by anvuew)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'dereverb_mel_band_roformer_anvuew.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt'),
            'download_urls': [
                'https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml',
                'https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt'
            ],
            'needs_conf_edit': True
        },
        'dereverb-echo_128_4_4 (by Sucial)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_dereverb-echo_128_4_4_mel_band_roformer.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'dereverb-echo_128_4_4_mel_band_roformer_sdr_dry_12.4235.ckpt'),
            'download_urls': [
                'https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb-echo_128_4_4_mel_band_roformer.yaml',
                'https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb-echo_128_4_4_mel_band_roformer_sdr_dry_12.4235.ckpt'
            ],
            'needs_conf_edit': True
        },
        'dereverb_echo_mbr_v2 (by Sucial)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_dereverb_echo_mbr_v2.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'dereverb_echo_mbr_v2_sdr_dry_13.4843.ckpt'),
            'download_urls': [
                'https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb_echo_mbr_v2.yaml',
                'https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb_echo_mbr_v2_sdr_dry_13.4843.ckpt'
            ],
            'needs_conf_edit': True
        },
        'de_big_reverb_mbr_ep_362 (by Sucial)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_dereverb_echo_mbr_v2.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'de_big_reverb_mbr_ep_362.ckpt'),
            'download_urls': [
                'https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb_echo_mbr_v2.yaml',
                'https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/de_big_reverb_mbr_ep_362.ckpt'
            ],
            'needs_conf_edit': True
        },
        'de_super_big_reverb_mbr_ep_346 (by Sucial)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_dereverb_echo_mbr_v2.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'de_super_big_reverb_mbr_ep_346.ckpt'),
            'download_urls': [
                'https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb_echo_mbr_v2.yaml',
                'https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/de_super_big_reverb_mbr_ep_346.ckpt'
            ],
            'needs_conf_edit': True
        },
        'dereverb_room (by anvuew)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'dereverb_room_anvuew.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'dereverb_room_anvuew_sdr_13.7432.ckpt'),
            'download_urls': [
                'https://huggingface.co/anvuew/dereverb_room/resolve/main/dereverb_room_anvuew.yaml',
                'https://huggingface.co/anvuew/dereverb_room/resolve/main/dereverb_room_anvuew_sdr_13.7432.ckpt'
            ],
            'needs_conf_edit': True
        }
    },
    "Karaoke": {
        'KARAOKE-MelBand-Roformer (by aufr33 & viperx)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_mel_band_roformer_karaoke.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt'),
            'download_urls': [
                'https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt',
                'https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/config_mel_band_roformer_karaoke.yaml'
            ],
            'needs_conf_edit': True
        },
        'KaraokeGabox (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'karaokegabox_1750911344.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'Karaoke_GaboxV1.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/karaoke/karaokegabox_1750911344.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/karaoke/Karaoke_GaboxV1.ckpt'
            ],
            'needs_conf_edit': True
        },
        'bs_karaoke_gabox_IS (by Gabox)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'karaoke_bs_roformer.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'bs_karaoke_gabox_IS.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/bsroformers/karaoke_bs_roformer.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/bsroformers/bs_karaoke_gabox_IS.ckpt'
            ],
            'needs_conf_edit': True
        },
        'bs_roformer_karaoke_frazer_becruily': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_karaoke_frazer_becruily.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'bs_roformer_karaoke_frazer_becruily.ckpt'),
            'download_urls': [
                'https://huggingface.co/becruily/bs-roformer-karaoke/resolve/main/config_karaoke_frazer_becruily.yaml',
                'https://huggingface.co/becruily/bs-roformer-karaoke/resolve/main/bs_roformer_karaoke_frazer_becruily.ckpt'
            ],
            'needs_conf_edit': True
        },
        'mel_band_roformer_karaoke_becruily': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_karaoke_becruily.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'mel_band_roformer_karaoke_becruily.ckpt'),
            'download_urls': [
                'https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/main/config_karaoke_becruily.yaml',
                'https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/main/mel_band_roformer_karaoke_becruily.ckpt'
            ],
            'needs_conf_edit': True
        }
    },   
    "Other Models": {
        'OTHER-BS-Roformer_1053 (by viperx)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'model_bs_roformer_ep_937_sdr_10.5309.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model_bs_roformer_ep_937_sdr_10.5309.ckpt'),
            'download_urls': [
                'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_937_sdr_10.5309.ckpt',
                'https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_937_sdr_10.5309.yaml'
            ],
            'needs_conf_edit': True
        },
        'CROWD-REMOVAL-MelBand-Roformer (by aufr33)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'model_mel_band_roformer_crowd.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt'),
            'download_urls': [
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt',
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/model_mel_band_roformer_crowd.yaml'
            ],
            'needs_conf_edit': True
        },
        'CINEMATIC-BandIt_Plus (by kwatcharasupat)': {
            'model_type': 'bandit',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_dnr_bandit_bsrnn_multi_mus64.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model_bandit_plus_dnr_sdr_11.47.chpt'),
            'download_urls': [
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/config_dnr_bandit_bsrnn_multi_mus64.yaml',
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/model_bandit_plus_dnr_sdr_11.47.chpt'
            ],
            'needs_conf_edit': False
        },
        'CINEMATIC-BandIt_v2 multi (by kwatcharasupat)': {
            'model_type': 'bandit_v2',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_dnr_bandit_v2_mus64.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'checkpoint-multi_state_dict.ckpt'),
            'download_urls': [
                'https://huggingface.co/jarredou/banditv2_state_dicts_only/resolve/main/checkpoint-multi_state_dict.ckpt',
                'https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/refs/heads/main/configs/config_dnr_bandit_v2_mus64.yaml'
            ],
            'needs_conf_edit': True
        },
        'DRUMSEP-MDX23C_DrumSep_6stem (by aufr33 & jarredou)': {
            'model_type': 'mdx23c',
            'config_path': os.path.join(CHECKPOINT_DIR, 'aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt'),
            'download_urls': [
                'https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt',
                'https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml'
            ],
            'needs_conf_edit': False
        },
        'SYH99999/MelBandRoformerSYHFTB1_Model1 (by Amane)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model.ckpt'),
            'download_urls': [
                'https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/config.yaml',
                'https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/model.ckpt'
            ],
            'needs_conf_edit': True
        },
        'SYH99999/MelBandRoformerSYHFTB1_Model2 (by Amane)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model2.ckpt'),
            'download_urls': [
                'https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/config.yaml',
                'https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/model2.ckpt'
            ],
            'needs_conf_edit': True
        },
        'SYH99999/MelBandRoformerSYHFTB1_Model3 (by Amane)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model3.ckpt'),
            'download_urls': [
                'https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/config.yaml',
                'https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/model3.ckpt'
            ],
            'needs_conf_edit': True
        },
        'bs_hyperace (by unwa)': {
            'model_type': 'bs_roformer_custom',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_hyperace.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'bs_hyperace.ckpt'),
            'download_urls': [
                ('https://huggingface.co/pcunwa/BS-Roformer-HyperACE/resolve/main/config.yaml', 'config_hyperace.yaml'),
                'https://huggingface.co/pcunwa/BS-Roformer-HyperACE/resolve/main/bs_hyperace.ckpt'
            ],
            'custom_model_url': 'https://huggingface.co/pcunwa/BS-Roformer-HyperACE/resolve/main/bs_roformer.py',
            'needs_conf_edit': True
        },
        'becruily_deux (by becruily)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_deux_becruily.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'becruily_deux.ckpt'),
            'download_urls': [
                'https://huggingface.co/becruily/mel-band-roformer-deux/resolve/main/config_deux_becruily.yaml',
                'https://huggingface.co/becruily/mel-band-roformer-deux/resolve/main/becruily_deux.ckpt'
            ],
            'needs_conf_edit': True
        },
        'becruily_guitar (by becruily)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_guitar_becruily.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'becruily_guitar.ckpt'),
            'download_urls': [
                'https://huggingface.co/becruily/mel-band-roformer-guitar/resolve/main/config_guitar_becruily.yaml',
                'https://huggingface.co/becruily/mel-band-roformer-guitar/resolve/main/becruily_guitar.ckpt'
            ],
            'needs_conf_edit': True
        },
        'aspiration_mel_band_roformer (by Sucial)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_aspiration_mel_band_roformer.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'aspiration_mel_band_roformer_sdr_18.9845.ckpt'),
            'download_urls': [
                'https://huggingface.co/Sucial/Aspiration_Mel_Band_Roformer/resolve/main/config_aspiration_mel_band_roformer.yaml',
                'https://huggingface.co/Sucial/Aspiration_Mel_Band_Roformer/resolve/main/aspiration_mel_band_roformer_sdr_18.9845.ckpt'
            ],
            'needs_conf_edit': True
        },
        'dereverb_echo_mbr_v2 (by Sucial)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_dereverb_echo_mbr_v2.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'dereverb_echo_mbr_v2_sdr_dry_13.4843.ckpt'),
            'download_urls': [
                'https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb_echo_mbr_v2.yaml',
                'https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb_echo_mbr_v2_sdr_dry_13.4843.ckpt'
            ],
            'needs_conf_edit': True
        },
        'mdx23c_similarity (by ZFTurbo)': {
            'model_type': 'mdx23c',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_mdx23c_similarity.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model_mdx23c_ep_271_l1_freq_72.2383.ckpt'),
            'download_urls': [
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.10/config_mdx23c_similarity.yaml',
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.10/model_mdx23c_ep_271_l1_freq_72.2383.ckpt'
            ],
            'needs_conf_edit': False
        },
        'mel_band_roformer_Lead_Rhythm_Guitar (by listra92)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_mel_band_roformer_Lead_Rhythm_Guitar.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model_mel_band_roformer_ep_72_sdr_3.2232.ckpt'),
            'download_urls': [
                'https://huggingface.co/listra92/MyModels/resolve/main/misc/config_mel_band_roformer_Lead_Rhythm_Guitar.yaml',
                'https://huggingface.co/listra92/MyModels/resolve/main/misc/model_mel_band_roformer_ep_72_sdr_3.2232.ckpt'
            ],
            'needs_conf_edit': True
        },
        'last_bs_roformer_4stem (by Amane)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_last_bs.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'last_bs_roformer.ckpt'),
            'download_urls': [
                'https://huggingface.co/listra92/MyModels/resolve/main/misc/config.yaml',
                'https://huggingface.co/listra92/MyModels/resolve/main/misc/last_bs_roformer.ckpt'
            ],
            'needs_conf_edit': True
        },
        'bs_roformer_4stems_ft (by SYH99999)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_bs_4stems_ft.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'bs_roformer_4stems_ft.pth'),
            'download_urls': [
                'https://huggingface.co/SYH99999/bs_roformer_4stems_ft/resolve/main/config.yaml',
                'https://huggingface.co/SYH99999/bs_roformer_4stems_ft/resolve/main/bs_roformer_4stems_ft.pth'
            ],
            'needs_conf_edit': True
        },
        'CINEMATIC-BandIt_v2_Eng (by kwatcharasupat)': {
            'model_type': 'bandit_v2',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_dnr_bandit_v2_mus64.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'checkpoint-eng_state_dict.ckpt'),
            'download_urls': [
                'https://huggingface.co/jarredou/banditv2_state_dicts_only/resolve/main/config_dnr_bandit_v2_mus64.yaml',
                'https://huggingface.co/jarredou/banditv2_state_dicts_only/resolve/main/checkpoint-eng_state_dict.ckpt'
            ],
            'needs_conf_edit': True
        }
    }
}

def get_model_config(clean_model=None, chunk_size=None, overlap=None):
    """Returns model type, config path, and checkpoint path for a given model name, downloading files if needed.
    
    download_urls can contain:
        - Simple strings: 'url' - downloads with filename from URL
        - Tuples: ('url', 'target_filename') - downloads with custom filename
    
    Also handles custom models loaded from custom_models.json
    """
    if clean_model is None:
        all_models = {model_name for category in MODEL_CONFIGS.values() for model_name in category.keys()}
        # Add custom models
        custom_models = load_custom_models()
        all_models.update(custom_models.keys())
        return all_models
    
    # First check built-in models
    for category in MODEL_CONFIGS.values():
        if clean_model in category:
            config = category[clean_model]
            for url_entry in config['download_urls']:
                # Handle both simple URL strings and (url, target_filename) tuples
                if isinstance(url_entry, tuple):
                    url, target_filename = url_entry
                    download_file(url, target_filename=target_filename)
                else:
                    download_file(url_entry)
            if config.get('custom_model_url'):
                custom_path = os.path.join(BASE_DIR, 'models', 'bs_roformer', 'bs_roformer_custom')
                os.makedirs(custom_path, exist_ok=True)
                # Create __init__.py for Python import support
                init_file = os.path.join(custom_path, '__init__.py')
                if not os.path.exists(init_file):
                    with open(init_file, 'w') as f:
                        f.write('# Auto-generated for custom BSRoformer models\n')
                download_file(config['custom_model_url'], path=custom_path)
            if config['needs_conf_edit'] and chunk_size is not None and overlap is not None:
                conf_edit(config['config_path'], chunk_size, overlap)
            return config['model_type'], config['config_path'], config['start_check_point']
    
    # Then check custom models
    custom_models = load_custom_models()
    if clean_model in custom_models:
        config = custom_models[clean_model]
        checkpoint_path = os.path.join(CHECKPOINT_DIR, config['checkpoint_filename'])
        config_path = os.path.join(CHECKPOINT_DIR, config['config_filename'])
        
        # Download checkpoint
        download_file(config['checkpoint_url'], target_filename=config['checkpoint_filename'])
        # Download config with custom filename
        download_file(config['config_url'], target_filename=config['config_filename'])
        
        # Handle custom model URL if present
        if config.get('custom_model_url'):
            custom_path = os.path.join(BASE_DIR, 'models', 'bs_roformer', 'bs_roformer_custom')
            os.makedirs(custom_path, exist_ok=True)
            init_file = os.path.join(custom_path, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write('# Auto-generated for custom BSRoformer models\n')
            download_file(config['custom_model_url'], path=custom_path)
        
        # Apply config edits if needed
        if config.get('needs_conf_edit', True) and chunk_size is not None and overlap is not None:
            conf_edit(config_path, chunk_size, overlap)
        
        return config['model_type'], config_path, checkpoint_path
    
    return "", "", ""

def get_all_model_configs_with_custom():
    """Returns MODEL_CONFIGS with Custom Models category added dynamically."""
    all_configs = dict(MODEL_CONFIGS)
    custom_models = load_custom_models()
    if custom_models:
        all_configs["Custom Models"] = {
            name: {
                'model_type': cfg['model_type'],
                'config_path': os.path.join(CHECKPOINT_DIR, cfg['config_filename']),
                'start_check_point': os.path.join(CHECKPOINT_DIR, cfg['checkpoint_filename']),
                'download_urls': [cfg['checkpoint_url'], cfg['config_url']],
                'custom_model_url': cfg.get('custom_model_url'),
                'needs_conf_edit': cfg.get('needs_conf_edit', True)
            }
            for name, cfg in custom_models.items()
        }
    return all_configs

get_model_config.keys = lambda: {model_name for category in MODEL_CONFIGS.values() for model_name in category.keys()}.union(load_custom_models().keys())
