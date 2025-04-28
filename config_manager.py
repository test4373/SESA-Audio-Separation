import os
import json
from pathlib import Path

# Define config directory in Google Drive
CONFIG_DIR = "/content/drive/MyDrive/SESA-Config"
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

def load_config():
    """Load configuration from config.json."""
    default_config = {
        "favorites": [],
        "settings": {
            "chunk_size": 352800,
            "overlap": 2,
            "export_format": "wav FLOAT",
            "use_tta": False,
            "use_demud_phaseremix_inst": False,
            "extract_instrumental": False,
            "use_apollo": True,
            "apollo_chunk_size": 19,
            "apollo_overlap": 2,
            "apollo_method": "normal_method",
            "apollo_normal_model": "Apollo Universal Model",
            "apollo_midside_model": "Apollo Universal Model",
            "use_matchering": False,
            "matchering_passes": 1,
            "model_category": "Vocal Models",
            "selected_model": None,
            "auto_category": "Vocal Models",
            "selected_models": [],
            "auto_ensemble_type": "avg_wave",
            "manual_ensemble_type": "avg_wave",
            "manual_weights": ""
        },
        "presets": {}
    }

    os.makedirs(CONFIG_DIR, exist_ok=True)
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2)
        return default_config

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        # Merge with default config to ensure all keys exist
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if subkey not in config[key]:
                        config[key][subkey] = subvalue
        return config
    except json.JSONDecodeError:
        print("Warning: config.json is corrupted. Creating a new one.")
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2)
        return default_config

def save_config(favorites, settings, presets):
    """Save configuration to config.json."""
    config = {
        "favorites": favorites,
        "settings": settings,
        "presets": presets
    }
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

def update_favorites(favorites, model, add=True):
    """Update favorites list."""
    cleaned_model = model
    new_favorites = favorites.copy()
    if add and cleaned_model not in new_favorites:
        new_favorites.append(cleaned_model)
    elif not add and cleaned_model in new_favorites:
        new_favorites.remove(cleaned_model)
    return new_favorites

def save_preset(presets, preset_name, models, ensemble_method, manual_ensemble_type, manual_weights, **kwargs):
    """Save a preset."""
    new_presets = presets.copy()
    cleaned_models = [clean_model(model) for model in models]
    new_presets[preset_name] = {
        "models": cleaned_models,
        "ensemble_method": ensemble_method,
        "manual_ensemble_type": manual_ensemble_type,
        "manual_weights": manual_weights,
        "chunk_size": kwargs.get("chunk_size", load_config()["settings"]["chunk_size"]),
        "overlap": kwargs.get("overlap", load_config()["settings"]["overlap"]),
        "use_tta": kwargs.get("use_tta", load_config()["settings"]["use_tta"]),
        "extract_instrumental": kwargs.get("extract_instrumental", load_config()["settings"]["extract_instrumental"]),
        "use_apollo": kwargs.get("use_apollo", load_config()["settings"]["use_apollo"]),
        "apollo_chunk_size": kwargs.get("apollo_chunk_size", load_config()["settings"]["apollo_chunk_size"]),
        "apollo_overlap": kwargs.get("apollo_overlap", load_config()["settings"]["apollo_overlap"]),
        "apollo_method": kwargs.get("apollo_method", load_config()["settings"]["apollo_method"]),
        "apollo_normal_model": kwargs.get("apollo_normal_model", load_config()["settings"]["apollo_normal_model"]),
        "apollo_midside_model": kwargs.get("apollo_midside_model", load_config()["settings"]["apollo_midside_model"]),
        "use_matchering": kwargs.get("use_matchering", load_config()["settings"]["use_matchering"]),
        "matchering_passes": kwargs.get("matchering_passes", load_config()["settings"]["matchering_passes"]),
        "auto_category": kwargs.get("auto_category", load_config()["settings"]["auto_category"])
    }
    return new_presets

def delete_preset(presets, preset_name):
    """Delete a preset."""
    new_presets = presets.copy()
    if preset_name in new_presets:
        del new_presets[preset_name]
    return new_presets

def clean_model(model):
    """Remove ⭐ from model name if present."""
    return model.replace(" ⭐", "") if isinstance(model, str) else model
