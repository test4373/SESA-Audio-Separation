import gradio as gr
import os
import glob
import subprocess
from pathlib import Path
from datetime import datetime
import json
import sys
import time
import random
from helpers import update_model_dropdown, handle_file_upload, clear_old_output, save_uploaded_file, update_file_list, clean_model
from download import download_callback
from model import get_model_config, MODEL_CONFIGS
from processing import process_audio, auto_ensemble_process, ensemble_audio_fn, refresh_auto_output
from assets.i18n.i18n import I18nAuto
from config_manager import load_config, save_config, update_favorites, save_preset, delete_preset
import logging
logging.basicConfig(filename='sesa_gui.log', level=logging.DEBUG)

# BASE_DIR tanımı
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "assets")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
URL_FILE = os.path.join(CONFIG_DIR, "last_url.txt")

# Load user config at startup
user_config = load_config()
initial_settings = user_config["settings"]
initial_favorites = user_config["favorites"]
initial_presets = user_config["presets"]

# Ensure auto_category is valid
if "auto_category" not in initial_settings or initial_settings["auto_category"] not in MODEL_CONFIGS:
    initial_settings["auto_category"] = "Vocal Models"

# Config dosyası yoksa oluştur
if not os.path.exists(CONFIG_FILE):
    default_config = {
        "lang": {"override": False, "selected_lang": "auto"},
        "sharing": {
            "method": "gradio",
            "ngrok_token": "",
            "port": random.randint(1000, 9000)  # Random port instead of fixed
        }
    }
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(default_config, f, indent=2)
else:  # If the file exists, load and update if necessary
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        # Ensure 'lang' key exists
        if "lang" not in config:
            config["lang"] = {"override": False, "selected_lang": "auto"}
        # Add 'sharing' key if it doesn't exist
        if "sharing" not in config:
            config["sharing"] = {
                "method": "gradio",
                "ngrok_token": "",
                "port": random.randint(1000, 9000)  # Random port instead of fixed
            }
        # Save the updated configuration
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except json.JSONDecodeError:  # Handle corrupted JSON
        print("Warning: config.json is corrupted. Creating a new one.")
        default_config = {
            "lang": {"override": False, "selected_lang": "auto"},
            "sharing": {
                "method": "gradio",
                "ngrok_token": "",
                "port": random.randint(1000, 9000)  # Random port instead of fixed
            }
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2)

# I18nAuto örneği (arayüz başlamadan önce dil yüklenir)
i18n = I18nAuto()

# Çıktı formatları
OUTPUT_FORMATS = ['wav', 'flac', 'mp3', 'ogg', 'opus', 'm4a', 'aiff', 'ac3']

# Arayüz oluşturma fonksiyonu
def create_interface():
    css = """
    body {
        background: linear-gradient(to bottom, rgba(45, 11, 11, 0.9), rgba(0, 0, 0, 0.8)), url('/content/logo.jpg') no-repeat center center fixed;
        background-size: cover;
        min-height: 100vh;
        margin: 0;
        padding: 1rem;
        font-family: 'Poppins', sans-serif;
        color: #C0C0C0;
        overflow-x: hidden;
    }
    .header-text {
        text-align: center;
        padding: 100px 20px 20px;
        color: #ff4040;
        font-size: 3rem;
        font-weight: 900;
        text-shadow: 0 0 10px rgba(255, 64, 64, 0.5);
        z-index: 1500;
        animation: text-glow 2s infinite;
    }
    .header-subtitle {
        text-align: center;
        color: #C0C0C0;
        font-size: 1.2rem;
        font-weight: 300;
        margin-top: -10px;
        text-shadow: 0 0 5px rgba(255, 64, 64, 0.3);
    }
    .gr-tab {
        background: rgba(128, 0, 0, 0.5) !important;
        border-radius: 12px 12px 0 0 !important;
        margin: 0 5px !important;
        color: #C0C0C0 !important;
        border: 1px solid #ff4040 !important;
        z-index: 1500;
        transition: background 0.3s ease, color 0.3s ease;
        padding: 10px 20px !important;
        font-size: 1.1rem !important;
    }
    button {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        background: #800000 !important;
        border: 1px solid #ff4040 !important;
        color: #C0C0C0 !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        box-shadow: 0 2px 10px rgba(255, 64, 64, 0.3);
    }
    button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 10px 40px rgba(255, 64, 64, 0.7) !important;
        background: #ff4040 !important;
    }
    .compact-upload.horizontal {
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
        max-width: 400px !important;
        height: 40px !important;
        padding: 0 12px !important;
        border: 1px solid #ff4040 !important;
        background: rgba(128, 0, 0, 0.5) !important;
        border-radius: 8px !important;
    }
    .compact-dropdown {
        --padding: 8px 12px !important;
        --radius: 10px !important;
        border: 1px solid #ff4040 !important;
        background: rgba(128, 0, 0, 0.5) !important;
        color: #C0C0C0 !important;
    }
    #custom-progress {
        margin-top: 10px;
        padding: 10px;
        background: rgba(128, 0, 0, 0.3);
        border-radius: 8px;
        border: 1px solid #ff4040;
    }
    #progress-bar {
        height: 20px;
        background: linear-gradient(to right, #6e8efb, #ff4040);
        border-radius: 5px;
        transition: width 0.5s ease-in-out;
        max-width: 100% !important;
    }
    .gr-accordion {
        background: rgba(128, 0, 0, 0.5) !important;
        border-radius: 10px !important;
        border: 1px solid #ff4040 !important;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #ff4040;
        font-size: 14px;
        margin-top: 40px;
        background: rgba(128, 0, 0, 0.3);
        border-top: 1px solid #ff4040;
    }
    #log-accordion {
        max-height: 400px;
        overflow-y: auto;
        background: rgba(0, 0, 0, 0.7) !important;
        padding: 10px;
        border-radius: 8px;
    }
    @keyframes text-glow {
        0% { text-shadow: 0 0 5px rgba(192, 192, 192, 0); }
        50% { text-shadow: 0 0 15px rgba(192, 192, 192, 1); }
        100% { text-shadow: 0 0 5px rgba(192, 192, 192, 0); }
    }
    """

    # Load user config at startup
    user_config = load_config()
    initial_settings = user_config["settings"]
    initial_favorites = user_config["favorites"]
    initial_presets = user_config["presets"]

    with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
        current_lang = gr.State(value=i18n.language)
        favorites_state = gr.State(value=initial_favorites)
        presets_state = gr.State(value=initial_presets)

        header_html = gr.HTML(
            value=f"""
            <div class="header-text">{i18n("SESA Audio Separation")}</div>
            <div class="header-subtitle">{i18n("ultimate_audio_separation")}</div>
            """
        )

        with gr.Tabs():
            with gr.Tab(i18n("audio_separation_tab"), id="separation_tab"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=380):
                        with gr.Accordion(i18n("input_model"), open=True) as input_model_accordion:
                            with gr.Tabs():
                                with gr.Tab(i18n("upload")) as upload_tab:
                                    input_audio_file = gr.File(
                                        file_types=[".wav", ".mp3", ".m4a", ".mp4", ".mkv", ".flac"],
                                        elem_classes=["compact-upload", "horizontal", "x-narrow"],
                                        label=""
                                    )
                                with gr.Tab(i18n("path")) as path_tab:
                                    file_path_input = gr.Textbox(placeholder=i18n("path_placeholder"))

                            with gr.Row():
                                model_category = gr.Dropdown(
                                    label=i18n("category"),
                                    choices=[i18n(cat) for cat in MODEL_CONFIGS.keys()],
                                    value=i18n(initial_settings["model_category"])
                                )
                                favorite_button = gr.Button(i18n("add_favorite"), variant="secondary", scale=0)

                            model_dropdown = gr.Dropdown(
                                label=i18n("model"),
                                choices=update_model_dropdown(i18n(initial_settings["model_category"]), favorites=initial_favorites)["choices"],
                                value=initial_settings["selected_model"]
                            )

                        with gr.Accordion(i18n("settings"), open=False) as settings_accordion:
                            with gr.Row():
                                with gr.Column(scale=1):
                                    export_format = gr.Dropdown(
                                        label=i18n("format"),
                                        choices=['wav FLOAT', 'flac PCM_16', 'flac PCM_24'],
                                        value=initial_settings["export_format"]
                                    )
                                with gr.Column(scale=1):
                                    chunk_size = gr.Dropdown(
                                        label=i18n("chunk_size"),
                                        choices=[352800, 485100],
                                        value=initial_settings["chunk_size"],
                                        info=i18n("chunk_size_info")
                                    )

                            with gr.Row():
                                with gr.Column(scale=2):
                                    overlap = gr.Slider(
                                        minimum=2,
                                        maximum=50,
                                        step=1,
                                        label=i18n("overlap"),
                                        value=initial_settings["overlap"],
                                        info=i18n("overlap_info")
                                    )

                            with gr.Accordion(i18n("backend_settings"), open=False) as backend_settings_accordion:
                                gr.Markdown(f"### {i18n('inference_backend')}")
                                with gr.Row():
                                    use_pytorch_optimized = gr.Checkbox(
                                        label="🔥 Optimized PyTorch",
                                        value=initial_settings.get("use_pytorch_optimized", False),
                                        info="Use optimized PyTorch backend with various optimizations"
                                    )
                                
                                with gr.Group(visible=initial_settings.get("use_pytorch_optimized", False)) as pytorch_opt_settings_group:
                                    optimize_mode = gr.Dropdown(
                                        label="Optimization Mode",
                                        choices=['default', 'compile', 'channels_last'],
                                        value=initial_settings.get("optimize_mode", "default"),
                                        info="compile requires PyTorch 2.0+, channels_last is good for modern GPUs"
                                    )

                            with gr.Row():
                                with gr.Column(scale=1):
                                    use_tta = gr.Checkbox(
                                        label=i18n("tta_boost"),
                                        info=i18n("tta_info"),
                                        value=initial_settings["use_tta"]
                                    )

                            with gr.Row():
                                with gr.Column(scale=1):
                                    use_demud_phaseremix_inst = gr.Checkbox(
                                        label=i18n("phase_fix"),
                                        info=i18n("phase_fix_info"),
                                        value=initial_settings["use_demud_phaseremix_inst"]
                                    )

                                with gr.Column(scale=1):
                                    extract_instrumental = gr.Checkbox(
                                        label=i18n("instrumental"),
                                        info=i18n("instrumental_info"),
                                        value=initial_settings["extract_instrumental"]
                                    )

                            with gr.Row():
                                use_apollo = gr.Checkbox(
                                    label=i18n("enhance_with_apollo"),
                                    value=initial_settings["use_apollo"],
                                    info=i18n("apollo_enhancement_info")
                                )

                            with gr.Group(visible=initial_settings["use_apollo"]) as apollo_settings_group:
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        apollo_chunk_size = gr.Slider(
                                            label=i18n("apollo_chunk_size"),
                                            minimum=3,
                                            maximum=25,
                                            step=1,
                                            value=initial_settings["apollo_chunk_size"],
                                            info=i18n("apollo_chunk_size_info"),
                                            interactive=True
                                        )
                                    with gr.Column(scale=1):
                                        apollo_overlap = gr.Slider(
                                            label=i18n("apollo_overlap"),
                                            minimum=2,
                                            maximum=10,
                                            step=1,
                                            value=initial_settings["apollo_overlap"],
                                            info=i18n("apollo_overlap_info"),
                                            interactive=True
                                        )

                                with gr.Row():
                                    apollo_method = gr.Dropdown(
                                        label=i18n("apollo_processing_method"),
                                        choices=[i18n("normal_method"), i18n("mid_side_method")],
                                        value=i18n(initial_settings["apollo_method"]),
                                        interactive=True
                                    )

                                with gr.Row(visible=initial_settings["apollo_method"] != "mid_side_method") as apollo_normal_model_row:
                                    apollo_normal_model = gr.Dropdown(
                                        label=i18n("apollo_normal_model"),
                                        choices=["MP3 Enhancer", "Lew Vocal Enhancer", "Lew Vocal Enhancer v2 (beta)", "Apollo Universal Model"],
                                        value=initial_settings["apollo_normal_model"],
                                        interactive=True
                                    )

                                with gr.Row(visible=initial_settings["apollo_method"] == "mid_side_method") as apollo_midside_model_row:
                                    apollo_midside_model = gr.Dropdown(
                                        label=i18n("apollo_mid_side_model"),
                                        choices=["MP3 Enhancer", "Lew Vocal Enhancer", "Lew Vocal Enhancer v2 (beta)", "Apollo Universal Model"],
                                        value=initial_settings["apollo_midside_model"],
                                        interactive=True
                                    )

                            with gr.Row():
                                use_matchering = gr.Checkbox(
                                    label=i18n("apply_matchering"),
                                    value=initial_settings.get("use_matchering", False),
                                    info=i18n("matchering_info")
                                )

                            with gr.Group(visible=initial_settings.get("use_matchering", True)) as matchering_settings_group:
                                matchering_passes = gr.Slider(
                                    label=i18n("matchering_passes"),
                                    minimum=1,
                                    maximum=5,
                                    step=1,
                                    value=initial_settings.get("matchering_passes", 1),
                                    info=i18n("matchering_passes_info"),
                                    interactive=True
                                )

                        with gr.Row():
                            process_btn = gr.Button(i18n("process"), variant="primary")
                            clear_old_output_btn = gr.Button(i18n("reset"), variant="secondary")
                        clear_old_output_status = gr.Textbox(label=i18n("status"), interactive=False)
                        
                        # Backend visibility handler
                        use_pytorch_optimized.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=use_pytorch_optimized,
                            outputs=pytorch_opt_settings_group
                        )

                        # Favorite handler
                        def update_favorite_button(model, favorites):
                            cleaned_model = clean_model(model) if model else None
                            is_favorited = cleaned_model in favorites if cleaned_model else False
                            return gr.update(value=i18n("remove_favorite") if is_favorited else i18n("add_favorite"))

                        def toggle_favorite(model, favorites):
                            if not model:
                                return favorites, gr.update(), gr.update()
                            cleaned_model = clean_model(model)
                            is_favorited = cleaned_model in favorites
                            new_favorites = update_favorites(favorites, cleaned_model, add=not is_favorited)
                            save_config(new_favorites, load_config()["settings"], load_config()["presets"])
                            category = model_category.value
                            return (
                                new_favorites,
                                gr.update(choices=update_model_dropdown(category, favorites=new_favorites)["choices"]),
                                gr.update(value=i18n("add_favorite") if is_favorited else i18n("remove_favorite"))
                            )

                        model_dropdown.change(
                            fn=update_favorite_button,
                            inputs=[model_dropdown, favorites_state],
                            outputs=favorite_button
                        )

                        favorite_button.click(
                            fn=toggle_favorite,
                            inputs=[model_dropdown, favorites_state],
                            outputs=[favorites_state, model_dropdown, favorite_button]
                        )

                        use_apollo.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=use_apollo,
                            outputs=apollo_settings_group
                        )

                        use_matchering.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=use_matchering,
                            outputs=matchering_settings_group
                        )

                        apollo_method.change(
                            fn=lambda x: [
                                gr.update(visible=x != i18n("mid_side_method")),
                                gr.update(visible=x == i18n("mid_side_method")),
                                "Apollo Universal Model" if x == i18n("mid_side_method") else None
                            ],
                            inputs=apollo_method,
                            outputs=[apollo_normal_model_row, apollo_midside_model_row, apollo_normal_model]
                        )

                    with gr.Column(scale=2, min_width=800):
                        with gr.Tabs():
                            with gr.Tab(i18n("main_tab")) as main_tab:
                                with gr.Column():
                                    original_audio = gr.Audio(label=i18n("original"), interactive=False)
                                    with gr.Row():
                                        vocals_audio = gr.Audio(label=i18n("vocals"), show_download_button=True)
                                        instrumental_audio = gr.Audio(label=i18n("instrumental_output"), show_download_button=True)
                                        other_audio = gr.Audio(label=i18n("other"), show_download_button=True)

                            with gr.Tab(i18n("details_tab")) as details_tab:
                                with gr.Column():
                                    with gr.Row():
                                        male_audio = gr.Audio(label=i18n("male"))
                                        female_audio = gr.Audio(label=i18n("female"))
                                        speech_audio = gr.Audio(label=i18n("speech"))
                                    with gr.Row():
                                        drum_audio = gr.Audio(label=i18n("drums"))
                                        bass_audio = gr.Audio(label=i18n("bass"))
                                    with gr.Row():
                                        effects_audio = gr.Audio(label=i18n("effects"))

                            with gr.Tab(i18n("advanced_tab")) as advanced_tab:
                                with gr.Column():
                                    with gr.Row():
                                        phaseremix_audio = gr.Audio(label=i18n("phase_remix"))
                                        dry_audio = gr.Audio(label=i18n("dry"))
                                    with gr.Row():
                                        music_audio = gr.Audio(label=i18n("music"))
                                        karaoke_audio = gr.Audio(label=i18n("karaoke"))
                                        bleed_audio = gr.Audio(label=i18n("bleed"))

                        separation_progress_html = gr.HTML(
                            value=f"""
                            <div id="custom-progress" style="margin-top: 10px;">
                                <div style="font-size: 1rem; color: #C0C0C0; margin-bottom: 5px;" id="progress-label">{i18n("waiting_for_processing")}</div>
                                <div style="width: 100%; background-color: #444; border-radius: 5px; overflow: hidden;">
                                    <div id="progress-bar" style="width: 0%; height: 20px; background-color: #6e8efb; transition: width 0.3s;"></div>
                                </div>
                            </div>
                            """
                        )
                        separation_process_status = gr.Textbox(
                            label=i18n("status"),
                            interactive=False,
                            placeholder=i18n("waiting_for_processing"),
                            visible=False
                        )
                        processing_tip = gr.Markdown(i18n("processing_tip"))

            with gr.Tab(i18n("auto_ensemble_tab"), id="auto_ensemble_tab"):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            auto_input_audio_file = gr.File(
                                file_types=[".wav", ".mp3", ".m4a", ".mp4", ".mkv", ".flac"],
                                label=i18n("upload_file")
                            )
                            auto_file_path_input = gr.Textbox(
                                label=i18n("enter_file_path"),
                                placeholder=i18n("file_path_placeholder"),
                                interactive=True
                            )

                        with gr.Accordion(i18n("advanced_settings"), open=False) as auto_settings_accordion:
                            with gr.Row():
                                auto_use_tta = gr.Checkbox(label=i18n("use_tta"), value=False)
                                auto_extract_instrumental = gr.Checkbox(label=i18n("instrumental_only"))

                            with gr.Row():
                                auto_overlap = gr.Slider(
                                    label=i18n("auto_overlap"),
                                    minimum=2,
                                    maximum=50,
                                    value=2,
                                    step=1
                                )
                                auto_chunk_size = gr.Dropdown(
                                    label=i18n("auto_chunk_size"),
                                    choices=[352800, 485100],
                                    value=352800
                                )
                                export_format2 = gr.Dropdown(
                                    label=i18n("output_format"),
                                    choices=['wav FLOAT', 'flac PCM_16', 'flac PCM_24'],
                                    value='wav FLOAT'
                                )

                            with gr.Row():
                                auto_use_apollo = gr.Checkbox(
                                    label=i18n("enhance_with_apollo"),
                                    value=False,
                                    info=i18n("apollo_enhancement_info")
                                )

                            with gr.Group(visible=False) as auto_apollo_settings_group:
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        auto_apollo_chunk_size = gr.Slider(
                                            label=i18n("apollo_chunk_size"),
                                            minimum=3,
                                            maximum=25,
                                            step=1,
                                            value=19,
                                            info=i18n("apollo_chunk_size_info"),
                                            interactive=True
                                        )
                                    with gr.Column(scale=1):
                                        auto_apollo_overlap = gr.Slider(
                                            label=i18n("apollo_overlap"),
                                            minimum=2,
                                            maximum=10,
                                            step=1,
                                            value=2,
                                            info=i18n("apollo_overlap_info"),
                                            interactive=True
                                        )

                                with gr.Row():
                                    auto_apollo_method = gr.Dropdown(
                                        label=i18n("apollo_processing_method"),
                                        choices=[i18n("normal_method"), i18n("mid_side_method")],
                                        value=i18n("normal_method"),
                                        interactive=True
                                    )

                                with gr.Row(visible=True) as auto_apollo_normal_model_row:
                                    auto_apollo_normal_model = gr.Dropdown(
                                        label=i18n("apollo_normal_model"),
                                        choices=["MP3 Enhancer", "Lew Vocal Enhancer", "Lew Vocal Enhancer v2 (beta)", "Apollo Universal Model"],
                                        value="Apollo Universal Model",
                                        interactive=True
                                    )

                                with gr.Row(visible=False) as auto_apollo_midside_model_row:
                                    auto_apollo_midside_model = gr.Dropdown(
                                        label=i18n("apollo_mid_side_model"),
                                        choices=["MP3 Enhancer", "Lew Vocal Enhancer", "Lew Vocal Enhancer v2 (beta)", "Apollo Universal Model"],
                                        value="Apollo Universal Model",
                                        interactive=True
                                    )

                            with gr.Row():
                                auto_use_matchering = gr.Checkbox(
                                    label=i18n("apply_matchering"),
                                    value=False,
                                    info=i18n("matchering_info")
                                )

                            with gr.Group(visible=True) as auto_matchering_settings_group:
                                auto_matchering_passes = gr.Slider(
                                    label=i18n("matchering_passes"),
                                    minimum=1,
                                    maximum=5,
                                    step=1,
                                    value=1,
                                    info=i18n("matchering_passes_info"),
                                    interactive=True
                                )

                        with gr.Group():
                            model_selection_header = gr.Markdown(f"### {i18n('model_selection')}")
                            with gr.Row():
                                auto_category_dropdown = gr.Dropdown(
                                    label=i18n("model_category"),
                                    choices=[i18n(cat) for cat in MODEL_CONFIGS.keys()],
                                    value=i18n("Vocal Models")
                                )
                                selected_models = gr.Dropdown(
                                    label=i18n("selected_models"),
                                    choices=update_model_dropdown(i18n(initial_settings["auto_category"]), favorites=initial_favorites)["choices"],
                                    value=initial_settings["selected_models"],
                                    multiselect=True
                                )

                            with gr.Row():
                                preset_dropdown = gr.Dropdown(
                                    label=i18n("select_preset"),
                                    choices=list(initial_presets.keys()),
                                    value=None,
                                    allow_custom_value=False,
                                    interactive=True
                                )
                            with gr.Row():
                                preset_name_input = gr.Textbox(
                                    label=i18n("preset_name"),
                                    placeholder=i18n("enter_preset_name"),
                                    interactive=True
                                )
                                save_preset_btn = gr.Button(i18n("save_preset"), variant="secondary", scale=0)
                                delete_preset_btn = gr.Button(i18n("delete_preset"), variant="secondary", scale=0)
                                refresh_presets_btn = gr.Button(i18n("refresh_presets"), variant="secondary", scale=0)

                        with gr.Group():
                            ensemble_settings_header = gr.Markdown(f"### {i18n('ensemble_settings')}")
                            with gr.Row():
                                auto_ensemble_type = gr.Dropdown(
                                    label=i18n("method"),
                                    choices=['avg_wave', 'median_wave', 'min_wave', 'max_wave',
                                             'avg_fft', 'median_fft', 'min_fft', 'max_fft'],
                                    value=initial_settings["auto_ensemble_type"]
                                )

                            ensemble_recommendation = gr.Markdown(i18n("recommendation"))

                        auto_process_btn = gr.Button(i18n("start_processing"), variant="primary")

                        def load_preset(preset_name, presets, category, favorites):
                            if preset_name and preset_name in presets:
                                preset = presets[preset_name]
                                # Mark starred models with ⭐
                                favorite_models = [f"{model} ⭐" if model in favorites else model for model in preset["models"]]
                                # Get the category from the preset, default to current category if not specified
                                preset_category = preset.get("auto_category_dropdown", category)
                                # Update model choices based on the preset's category
                                model_choices = update_model_dropdown(preset_category, favorites=favorites)["choices"]
                                print(f"Preset '{preset_name}' loaded with models: {favorite_models}, category: {preset_category}")
                                return (
                                    gr.update(value=preset_category),  # Update auto_category_dropdown
                                    gr.update(choices=model_choices, value=favorite_models),  # Update selected_models
                                    gr.update(value=preset["ensemble_method"])  # Update auto_ensemble_type
                                )
                            print(f"Preset '{preset_name}' not found.")
                            return gr.update(), gr.update(), gr.update()

                        def sync_presets():
                            """Reload presets from config and update dropdown."""
                            config = load_config()
                            return config["presets"], gr.update(choices=list(config["presets"].keys()), value=None)

                        preset_dropdown.change(
                            fn=load_preset,
                            inputs=[preset_dropdown, presets_state, auto_category_dropdown, favorites_state],
                            outputs=[auto_category_dropdown, selected_models, auto_ensemble_type]
                        )

                        def handle_save_preset(preset_name, models, ensemble_method, presets, favorites, auto_category_dropdown):
                            if not preset_name:
                                return gr.update(), presets, i18n("no_preset_name_provided")
                            if not models and not favorites:
                                return gr.update(), presets, i18n("no_models_selected_for_preset")
                            new_presets = save_preset(
                                presets, 
                                preset_name, 
                                models, 
                                ensemble_method,
                                auto_category_dropdown=auto_category_dropdown  # Pass the category explicitly
                            )
                            save_config(favorites, load_config()["settings"], new_presets)
                            print(f"Preset dropdown updated with choices: {list(new_presets.keys())}")
                            return gr.update(choices=list(new_presets.keys()), value=None), new_presets, i18n("preset_saved").format(preset_name)

                        save_preset_btn.click(
                            fn=handle_save_preset,
                            inputs=[preset_name_input, selected_models, auto_ensemble_type, presets_state, favorites_state, auto_category_dropdown],
                            outputs=[preset_dropdown, presets_state]
                        )

                        def handle_delete_preset(preset_name, presets):
                            if not preset_name or preset_name not in presets:
                                return gr.update(), presets
                            new_presets = delete_preset(presets, preset_name)
                            save_config(load_config()["favorites"], load_config()["settings"], new_presets)
                            return gr.update(choices=list(new_presets.keys()), value=None), new_presets

                        delete_preset_btn.click(
                            fn=handle_delete_preset,
                            inputs=[preset_dropdown, presets_state],
                            outputs=[preset_dropdown, presets_state]
                        )

                        refresh_presets_btn.click(
                            fn=sync_presets,
                            inputs=[],
                            outputs=[presets_state, preset_dropdown]
                        )

                        auto_use_apollo.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=auto_use_apollo,
                            outputs=auto_apollo_settings_group
                        )

                        auto_use_matchering.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=auto_use_matchering,
                            outputs=auto_matchering_settings_group
                        )

                        auto_apollo_method.change(
                            fn=lambda x: [
                                gr.update(visible=x != i18n("mid_side_method")),
                                gr.update(visible=x == i18n("mid_side_method")),
                                "Apollo Universal Model" if x == i18n("mid_side_method") else None
                            ],
                            inputs=auto_apollo_method,
                            outputs=[auto_apollo_normal_model_row, auto_apollo_midside_model_row, auto_apollo_normal_model]
                        )

                    with gr.Column():
                        with gr.Tabs():
                            with gr.Tab(i18n("original_audio_tab")) as original_audio_tab:
                                original_audio2 = gr.Audio(
                                    label=i18n("original_audio"),
                                    interactive=False,
                                    every=1,
                                    elem_id="original_audio_player"
                                )
                            with gr.Tab(i18n("ensemble_result_tab")) as ensemble_result_tab:
                                auto_output_audio = gr.Audio(
                                    label=i18n("output_preview"),
                                    show_download_button=True,
                                    interactive=False
                                )
                                refresh_output_btn = gr.Button(i18n("refresh_output"), variant="secondary")

                        ensemble_progress_html = gr.HTML(
                            value=f"""
                            <div id="custom-progress" style="margin-top: 10px;">
                                <div style="font-size: 1rem; color: #C0C0C0; margin-bottom: 5px;" id="progress-label">{i18n("waiting_for_processing")}</div>
                                <div style="width: 100%; background-color: #444; border-radius: 5px; overflow: hidden;">
                                    <div id="progress-bar" style="width: 0%; height: 20px; background-color: #6e8efb; transition: width 0.3s;"></div>
                                </div>
                            </div>
                            """
                        )
                        ensemble_process_status = gr.Textbox(
                            label=i18n("status"),
                            interactive=False,
                            placeholder=i18n("waiting_for_processing"),
                            visible=False
                        )
                        
            with gr.Tab(i18n("download_sources_tab"), id="download_tab"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(f"### {i18n('direct_links')}")
                        direct_url_input = gr.Textbox(label=i18n("audio_file_url"))
                        direct_download_btn = gr.Button(i18n("download_from_url"), variant="secondary")
                        direct_download_status = gr.Textbox(label=i18n("download_status"))
                        direct_download_output = gr.File(label=i18n("downloaded_file"), interactive=False)

                    with gr.Column():
                        gr.Markdown(f"### {i18n('cookie_management')}")
                        cookie_file = gr.File(
                            label=i18n("upload_cookies_txt"),
                            file_types=[".txt"],
                            interactive=True,
                            elem_id="cookie_upload"
                        )
                        cookie_info = gr.Markdown(i18n("cookie_info"))

            with gr.Tab(i18n("manual_ensemble_tab"), id="manual_ensemble_tab"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=400):
                        with gr.Accordion(i18n("input_sources"), open=True) as input_sources_accordion:
                            with gr.Row():
                                refresh_btn = gr.Button(i18n("refresh"), variant="secondary", size="sm")
                                ensemble_type = gr.Dropdown(
                                    label=i18n("ensemble_algorithm"),
                                    choices=['avg_wave', 'median_wave', 'min_wave', 'max_wave',
                                             'avg_fft', 'median_fft', 'min_fft', 'max_fft'],
                                    value='avg_wave'
                                )

                            file_dropdown_header = gr.Markdown(f"### {i18n('select_audio_files')}")
                            file_path = os.path.join(Path.home(), 'Music-Source-Separation', 'output')
                            initial_files = glob.glob(f"{file_path}/*.wav") + glob.glob(os.path.join(BASE_DIR, 'Music-Source-Separation-Training', 'old_output', '*.wav'))
                            file_dropdown = gr.Dropdown(
                                choices=initial_files,
                                label=i18n("available_files"),
                                multiselect=True,
                                interactive=True,
                                elem_id="file-dropdown"
                            )
                            weights_input = gr.Textbox(
                                label=i18n("custom_weights"),
                                placeholder=i18n("custom_weights_placeholder"),
                                info=i18n("custom_weights_info")
                            )

                    with gr.Column(scale=2, min_width=800):
                        with gr.Tabs():
                            with gr.Tab(i18n("result_preview_tab")) as result_preview_tab:
                                ensemble_output_audio = gr.Audio(
                                    label=i18n("ensembled_output"),
                                    interactive=False,
                                    show_download_button=True,
                                    elem_id="output-audio"
                                )
                            with gr.Tab(i18n("processing_log_tab")) as processing_log_tab:
                                with gr.Accordion(i18n("processing_details"), open=True, elem_id="log-accordion"):
                                    ensemble_status = gr.Textbox(
                                        label="",
                                        interactive=False,
                                        placeholder=i18n("processing_log_placeholder"),
                                        lines=10,
                                        max_lines=20,
                                        elem_id="log-box"
                                    )
                        with gr.Row():
                            ensemble_process_btn = gr.Button(
                                i18n("process_ensemble"),
                                variant="primary",
                                size="sm",
                                elem_id="process-btn"
                            )

        def save_settings_on_process(*args):
            apollo_method_value = args[13]
            backend_apollo_method = "mid_side_method" if apollo_method_value == i18n("mid_side_method") else "normal_method"
            cleaned_model = clean_model(args[1]) if args[1] else None
            settings = {
                "chunk_size": args[2],
                "overlap": args[3],
                "export_format": args[4],
                "use_pytorch_optimized": args[5],
                "optimize_mode": args[6],
                "use_tta": args[7],
                "use_demud_phaseremix_inst": args[8],
                "extract_instrumental": args[9],
                "use_apollo": args[10],
                "apollo_chunk_size": args[11],
                "apollo_overlap": args[12],
                "apollo_method": backend_apollo_method,
                "apollo_normal_model": args[14],
                "apollo_midside_model": args[15],
                "use_matchering": args[16],
                "matchering_passes": args[17],
                "model_category": args[18],
                "selected_model": cleaned_model,
                "auto_ensemble_type": args[9]
            }
            save_config(load_config()["favorites"], settings, load_config()["presets"])
            modified_args = list(args)
            modified_args[1] = cleaned_model
            modified_args[19] = cleaned_model
            return process_audio(*modified_args)

        def save_auto_ensemble_settings(*args):
            settings = load_config()["settings"]
            settings["auto_ensemble_type"] = args[7]
            settings["use_matchering"] = args[14]
            settings["matchering_passes"] = args[15]
            save_config(load_config()["favorites"], settings, load_config()["presets"])
            # Handle generator output from auto_ensemble_process
            output_audio, status, progress_html = None, i18n("waiting_for_processing"), ensemble_progress_html.value
            for update in auto_ensemble_process(*args):
                if isinstance(update, tuple) and len(update) == 3:
                    output_audio, status, progress_html = update
            return output_audio, status, progress_html

        def update_category_dropdowns(cat):
            logging.debug(f"Input category: {cat}")
            eng_cat = next((k for k in MODEL_CONFIGS.keys() if i18n(k) == cat), list(MODEL_CONFIGS.keys())[0])
            logging.debug(f"Using English category: {eng_cat}")
            choices = update_model_dropdown(eng_cat, favorites=load_config()["favorites"])["choices"]
            logging.debug(f"Model choices: {choices}")
            return gr.update(choices=choices), gr.update(choices=choices)

        model_category.change(
            fn=update_category_dropdowns,
            inputs=model_category,
            outputs=[model_dropdown, selected_models]
        )

        clear_old_output_btn.click(fn=clear_old_output, outputs=clear_old_output_status)

        input_audio_file.upload(
            fn=lambda x, y: handle_file_upload(x, y, is_auto_ensemble=False),
            inputs=[input_audio_file, file_path_input],
            outputs=[input_audio_file, original_audio]
        )
        file_path_input.change(
            fn=lambda x, y: handle_file_upload(x, y, is_auto_ensemble=False),
            inputs=[input_audio_file, file_path_input],
            outputs=[input_audio_file, original_audio]
        )

        auto_input_audio_file.upload(
            fn=lambda x, y: handle_file_upload(x, y, is_auto_ensemble=True),
            inputs=[auto_input_audio_file, auto_file_path_input],
            outputs=[auto_input_audio_file, original_audio2]
        )
        auto_file_path_input.change(
            fn=lambda x, y: handle_file_upload(x, y, is_auto_ensemble=True),
            inputs=[auto_input_audio_file, auto_file_path_input],
            outputs=[auto_input_audio_file, original_audio2]
        )

        auto_category_dropdown.change(
            fn=lambda cat: gr.update(choices=update_model_dropdown(next((k for k in MODEL_CONFIGS.keys() if i18n(k) == cat), list(MODEL_CONFIGS.keys())[0]), favorites=load_config()["favorites"])["choices"]),
            inputs=auto_category_dropdown,
            outputs=selected_models
        )

        def debug_inputs(*args):
            input_names = [
                "input_audio_file", "model_dropdown", "chunk_size", "overlap", "export_format",
                "use_pytorch_optimized", "optimize_mode",
                "use_tta", "use_demud_phaseremix_inst", "extract_instrumental",
                "use_apollo", "apollo_chunk_size", "apollo_overlap",
                "apollo_method", "apollo_normal_model", "apollo_midside_model",
                "use_matchering", "matchering_passes", "model_category", "selected_model"
            ]
            cleaned_args = list(args)
            cleaned_args[1] = clean_model(cleaned_args[1]) if cleaned_args[1] else None
            cleaned_args[19] = clean_model(cleaned_args[19]) if cleaned_args[19] else None
            for name, value in zip(input_names, cleaned_args):
                print(f"UI Input - {name}: {value}")
            return args

        process_btn.click(
            fn=lambda *args: save_settings_on_process(*debug_inputs(*args)),
            inputs=[
                input_audio_file, model_dropdown, chunk_size, overlap, export_format,
                use_pytorch_optimized, optimize_mode,
                use_tta, use_demud_phaseremix_inst, extract_instrumental,
                use_apollo, apollo_chunk_size, apollo_overlap,
                apollo_method, apollo_normal_model, apollo_midside_model,
                use_matchering, matchering_passes, model_category, model_dropdown
            ],
            outputs=[
                vocals_audio, instrumental_audio, phaseremix_audio, drum_audio, karaoke_audio,
                other_audio, bass_audio, effects_audio, speech_audio, bleed_audio, music_audio,
                dry_audio, male_audio, female_audio,
                separation_process_status, separation_progress_html
            ]
        )

        auto_process_btn.click(
            fn=save_auto_ensemble_settings,
            inputs=[
                auto_input_audio_file,
                selected_models,
                auto_chunk_size,
                auto_overlap,
                export_format2,
                auto_use_tta,
                auto_extract_instrumental,
                auto_ensemble_type,
                gr.State(None),
                auto_use_apollo,
                auto_apollo_normal_model,
                auto_apollo_chunk_size,
                auto_apollo_overlap,
                auto_apollo_method,
                auto_use_matchering,
                auto_matchering_passes,
                auto_apollo_midside_model
            ],
            outputs=[auto_output_audio, ensemble_process_status, ensemble_progress_html]
        )

        direct_download_btn.click(
            fn=download_callback,
            inputs=[direct_url_input, gr.State('direct'), cookie_file],
            outputs=[direct_download_output, direct_download_status, input_audio_file, auto_input_audio_file, original_audio, original_audio2]
        )

        refresh_output_btn.click(
            fn=refresh_auto_output,
            inputs=[],
            outputs=[auto_output_audio, ensemble_process_status]
        )

        refresh_btn.click(fn=update_file_list, outputs=file_dropdown)
        ensemble_process_btn.click(fn=ensemble_audio_fn, inputs=[file_dropdown, ensemble_type, weights_input], outputs=[ensemble_output_audio, ensemble_status])

        return demo
