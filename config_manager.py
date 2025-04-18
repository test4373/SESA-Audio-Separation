import os
import json
from pathlib import Path
import logging

# Set up logging (errors only)
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Config path in Google Drive
CONFIG_PATH = "/content/drive/MyDrive/SESA-Audio-Separation/user_config.json"

def clean_model(model):
    """Remove ⭐ prefix from model name."""
    return model.replace("⭐ ", "") if model.startswith("⭐ ") else model

def load_config():
    """Load user settings, favorites, and presets from config file, return defaults if file is missing."""
    default_config = {
        "favorites": [],
        "settings": {
            "chunk_size": 352800,
            "overlap": 2,
            "export_format": "wav FLOAT",
            "use_tta": False,
            "use_demud_phaseremix_inst": False,
            "extract_instrumental": False,
            "use_apollo": False,
            "apollo_chunk_size": 19,
            "apollo_overlap": 2,
            "apollo_method": "normal_method",
            "apollo_normal_model": "Apollo Universal Model",
            "apollo_midside_model": "Apollo Universal Model",
            "model_category": "Vocal Models",
            "selected_model": None,
            "auto_ensemble_type": "avg_wave"
        },
        "presets": {}
    }
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
            # Merge with defaults to handle missing keys
            for key in default_config:
                if key not in config:
                    config[key] = default_config[key]
                elif isinstance(default_config[key], dict):
                    for subkey in default_config[key]:
                        if subkey not in config[key]:
                            config[key][subkey] = default_config[key][subkey]
            return config
        else:
            logger.error(f"Config file not found at {CONFIG_PATH}. Using defaults.")
    except Exception as e:
        logger.error(f"Error loading user_config: {e}. Using defaults.")
    return default_config

def save_config(favorites, settings, presets):
    """Save favorites, settings, and presets to config file."""
    config = {
        "favorites": favorites,
        "settings": settings,
        "presets": presets
    }
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving user_config: {e}")

def update_favorites(current_favorites, model, add=True):
    """Add or remove a model from favorites."""
    new_favorites = current_favorites.copy()
    if add and model not in new_favorites:
        new_favorites.append(model)
    elif not add and model in new_favorites:
        new_favorites.remove(model)
    return new_favorites

def save_preset(presets, preset_name, models, ensemble_method):
    """Save a new preset or update an existing one, cleaning model names."""
    new_presets = presets.copy()
    new_presets[preset_name] = {
        "models": [clean_model(model) for model in models],
        "ensemble_method": ensemble_method
    }
    return new_presets

def delete_preset(presets, preset_name):
    """Delete a preset by name."""
    new_presets = presets.copy()
    new_presets.pop(preset_name, None)
    return new_presets
