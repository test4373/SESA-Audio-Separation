import gradio as gr
import os
import glob
import subprocess
from pathlib import Path
from datetime import datetime
import json
import sys
import time
from helpers import update_model_dropdown, handle_file_upload, clear_old_output, save_uploaded_file, update_file_list
from download import download_callback
from model import get_model_config, MODEL_CONFIGS
from processing import process_audio, auto_ensemble_process, ensemble_audio_fn, refresh_auto_output, copy_ensemble_to_drive, copy_to_drive
from assets.i18n.i18n import I18nAuto

# BASE_DIR tanımı
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "assets")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
URL_FILE = os.path.join(CONFIG_DIR, "last_url.txt")

# Config dosyası yoksa oluştur
if not os.path.exists(CONFIG_FILE):
    default_config = {"lang": {"override": False, "selected_lang": "auto"}}
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(default_config, f, indent=2)
    print(f"Created config.json at: {CONFIG_FILE}")

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

    with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
        current_lang = gr.State(value=i18n.language)

        header_html = gr.HTML(
            value=f"""
            <div class="header-text">{i18n("gecekondu_production")}</div>
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
                                    value=i18n("Vocal Models")
                                )
                                model_dropdown = gr.Dropdown(
                                    label=i18n("model"),
                                    choices=list(MODEL_CONFIGS["Vocal Models"].keys())
                                )

                        with gr.Accordion(i18n("settings"), open=False) as settings_accordion:
                            with gr.Row():
                                with gr.Column(scale=1):
                                    export_format = gr.Dropdown(
                                        label=i18n("format"),
                                        choices=['wav FLOAT', 'flac PCM_16', 'flac PCM_24'],
                                        value='wav FLOAT'
                                    )
                                with gr.Column(scale=1):
                                    chunk_size = gr.Dropdown(
                                        label=i18n("chunk_size"),
                                        choices=[352800, 485100],
                                        value=352800,
                                        info=i18n("chunk_size_info")
                                    )

                            with gr.Row():
                                with gr.Column(scale=2):
                                    overlap = gr.Slider(
                                        minimum=2,
                                        maximum=50,
                                        step=1,
                                        label=i18n("overlap"),
                                        value=2,
                                        info=i18n("overlap_info")  # Bu zaten doğru, ancak görünmüyorsa aşağıda Markdown ile ekleyeceğiz
                                    )

                            with gr.Row():
                                with gr.Column(scale=1):
                                    use_tta = gr.Checkbox(
                                        label=i18n("tta_boost"),
                                        info=i18n("tta_info")
                                    )
                                    
                            with gr.Row():
                                with gr.Column(scale=1):
                                    use_demud_phaseremix_inst = gr.Checkbox(
                                        label=i18n("phase_fix"),
                                        info=i18n("phase_fix_info")
                                    )

                                with gr.Column(scale=1):
                                    extract_instrumental = gr.Checkbox(
                                        label=i18n("instrumental"),
                                        info=i18n("instrumental_info")
                                    )

                            # Apollo ayarları
                            with gr.Row():
                                use_apollo = gr.Checkbox(
                                    label=i18n("enhance_with_apollo"),
                                    value=False,
                                    info=i18n("apollo_enhancement_info")
                                )

                            # Apollo ayarlarını tek bir grup altında topluyoruz
                            with gr.Group(visible=False) as apollo_settings_group:
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        apollo_chunk_size = gr.Slider(
                                            label=i18n("apollo_chunk_size"),
                                            minimum=3,
                                            maximum=25,
                                            step=1,
                                            value=19,
                                            info=i18n("apollo_chunk_size_info"),
                                            interactive=True
                                        )
                                    with gr.Column(scale=1):
                                        apollo_overlap = gr.Slider(
                                            label=i18n("apollo_overlap"),
                                            minimum=2,
                                            maximum=10,
                                            step=1,
                                            value=2,
                                            info=i18n("apollo_overlap_info"),
                                            interactive=True
                                        )

                                with gr.Row():
                                    apollo_method = gr.Dropdown(
                                        label=i18n("apollo_processing_method"),
                                        choices=[i18n("normal_method"), i18n("mid_side_method")],
                                        value=i18n("normal_method"),
                                        interactive=True
                                    )

                                with gr.Row(visible=True) as apollo_normal_model_row:
                                    apollo_normal_model = gr.Dropdown(
                                        label=i18n("apollo_normal_model"),
                                        choices=["MP3 Enhancer", "Lew Vocal Enhancer", "Lew Vocal Enhancer v2 (beta)", "Apollo Universal Model"],
                                        value="Apollo Universal Model",
                                        interactive=True
                                    )

                                with gr.Row(visible=False) as apollo_midside_model_row:
                                    apollo_midside_model = gr.Dropdown(
                                        label=i18n("apollo_mid_side_model"),
                                        choices=["MP3 Enhancer", "Lew Vocal Enhancer", "Lew Vocal Enhancer v2 (beta)", "Apollo Universal Model"],
                                        value="Apollo Universal Model",
                                        interactive=True
                                    )

                        with gr.Row():
                            process_btn = gr.Button(i18n("process"), variant="primary")
                            clear_old_output_btn = gr.Button(i18n("reset"), variant="secondary")
                        clear_old_output_status = gr.Textbox(label=i18n("status"), interactive=False)

                        # Apollo ayarlarının görünürlüğünü kontrol eden event
                        use_apollo.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=use_apollo,
                            outputs=apollo_settings_group
                        )

                        # Mid/Side model dropdown’ını kontrol eden event
                        apollo_method.change(
                            fn=lambda x: [gr.update(visible=x != i18n("mid_side_method")), gr.update(visible=x == i18n("mid_side_method"))],
                            inputs=apollo_method,
                            outputs=[apollo_normal_model_row, apollo_midside_model_row]
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
                        copy_to_drive_btn = gr.Button(i18n("copy_to_drive"), variant="secondary")
                        copy_status = gr.Textbox(
                            label=i18n("copy_status"),
                            interactive=False,
                            placeholder=i18n("files_will_be_copied"),
                            visible=True
                        )
                        processing_tip = gr.Markdown(i18n("processing_tip"))

            with gr.Tab(i18n("auto_ensemble_tab"), id="auto_ensemble_tab"):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            auto_input_audio_file = gr.File(label=i18n("upload_file"))
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

                        with gr.Group():
                            model_selection_header = gr.Markdown(f"### {i18n('model_selection')}")
                            with gr.Row():
                                auto_category_dropdown = gr.Dropdown(
                                    label=i18n("model_category"),
                                    choices=[i18n(cat) for cat in MODEL_CONFIGS.keys()],
                                    value=i18n("Vocal Models")
                                )

                                selected_models = gr.Dropdown(
                                    label=i18n("select_models"),
                                    choices=list(MODEL_CONFIGS["Vocal Models"].keys()),
                                    multiselect=True,
                                    max_choices=50,
                                    interactive=True
                                )

                        with gr.Group():
                            ensemble_settings_header = gr.Markdown(f"### {i18n('ensemble_settings')}")
                            with gr.Row():
                                auto_ensemble_type = gr.Dropdown(
                                    label=i18n("method"),
                                    choices=['avg_wave', 'median_wave', 'min_wave', 'max_wave',
                                             'avg_fft', 'median_fft', 'min_fft', 'max_fft'],
                                    value='avg_wave'
                                )

                            ensemble_recommendation = gr.Markdown(i18n("recommendation"))

                        auto_process_btn = gr.Button(i18n("start_processing"), variant="primary")

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
                        ensemble_copy_to_drive_btn = gr.Button(i18n("copy_to_drive"), variant="secondary")
                        ensemble_copy_status = gr.Textbox(
                            label=i18n("ensemble_copy_status"),
                            interactive=False,
                            placeholder=i18n("ensemble_copy_status"),
                            visible=True
                        )
                        guidelines = gr.Markdown(i18n("guidelines"))

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

        footer_html = gr.HTML(
            value=f"""
            <div class="footer">
                <div>{i18n("presented_by")}</div>
                <div style="margin-top: 10px;"></div>
            </div>
            """
        )

        # Event handlers
        model_category.change(fn=update_model_dropdown, inputs=model_category, outputs=model_dropdown)
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

        auto_category_dropdown.change(fn=update_model_dropdown, inputs=auto_category_dropdown, outputs=selected_models)

        process_btn.click(
            fn=process_audio,
            inputs=[
                input_audio_file, model_dropdown, chunk_size, overlap, export_format,
                use_tta, use_demud_phaseremix_inst, extract_instrumental, model_dropdown,
                use_apollo, apollo_chunk_size, apollo_overlap,
                apollo_method, apollo_normal_model, apollo_midside_model
            ],
            outputs=[
                vocals_audio, instrumental_audio, phaseremix_audio, drum_audio, karaoke_audio,
                other_audio, bass_audio, effects_audio, speech_audio, bleed_audio, music_audio,
                dry_audio, male_audio, female_audio,
                separation_process_status, separation_progress_html
            ]
        )

        auto_process_btn.click(
            fn=auto_ensemble_process,
            inputs=[
                auto_input_audio_file, selected_models, auto_chunk_size, auto_overlap, export_format2,
                auto_use_tta, auto_extract_instrumental, auto_ensemble_type, gr.State(None)
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

        ensemble_copy_to_drive_btn.click(
            fn=copy_ensemble_to_drive,
            inputs=[],
            outputs=[ensemble_copy_status]
        )

        copy_to_drive_btn.click(
            fn=copy_to_drive,
            inputs=[],
            outputs=[copy_status]
        )

        refresh_btn.click(fn=update_file_list, outputs=file_dropdown)
        ensemble_process_btn.click(fn=ensemble_audio_fn, inputs=[file_dropdown, ensemble_type, weights_input], outputs=[ensemble_output_audio, ensemble_status])

        return demo
