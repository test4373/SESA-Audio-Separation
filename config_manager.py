from helpers import clean_model
import json
import os
import logging

# Set up logging (errors only)
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Config path in Google Drive
CONFIG_FILE = "/content/drive/MyDrive/SESA-Audio-Separation/user_config.json"

def load_config():
    default_config = {
        "favorites": [],
        "presets": {},
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
            "auto_ensemble_type": "avg_wave",
            "auto_category": "Vocal Models",  # Already added from previous fix
            "selected_models": []  # Add this line
        }
    }
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
                # Ensure all default keys exist
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                # Ensure all settings keys exist
                for key, value in default_config["settings"].items():
                    if key not in config["settings"]:
                        config["settings"][key] = value
                return config
        return default_config
    except Exception as e:
        print(f"Error loading config: {e}")
        return default_config

def save_config(favorites, settings, presets):
    config = {
        "favorites": favorites,
        "settings": settings,
        "presets": presets
    }
    try:
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print("Config saved successfully.")
    except Exception as e:
        print(f"Error saving config: {e}")

def update_favorites(favorites, model, add=True):
    cleaned_model = clean_model(model)
    new_favorites = favorites.copy()
    if add and cleaned_model not in new_favorites:
        new_favorites.append(cleaned_model)
    elif not add and cleaned_model in new_favorites:
        new_favorites.remove(cleaned_model)
    return new_favorites

def save_preset(presets, preset_name, models, ensemble_method):
    if not preset_name:
        print("Preset name cannot be empty.")
        return presets
    if not models:
        print("No models selected.")
        return presets
    
    # Include starred models from favorites
    config = load_config()
    favorites = config["favorites"]
    # Combine selected models and favorites, remove duplicates
    combined_models = list(set([clean_model(model) for model in models] + favorites))
    
    new_presets = presets.copy()
    new_presets[preset_name] = {
        "models": combined_models,
        "ensemble_method": ensemble_method
    }
    print(f"Preset '{preset_name}' saved with models: {combined_models}, method: {ensemble_method}")
    return new_presets

def delete_preset(presets, preset_name):
    new_presets = presets.copy()
    if preset_name in new_presets:
        del new_presets[preset_name]
        print(f"Preset '{preset_name}' deleted.")
    return new_presets
