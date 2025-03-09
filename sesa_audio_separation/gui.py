import gradio as gr
import os
import glob
import subprocess
from pathlib import Path
from datetime import datetime
from helpers import update_model_dropdown, handle_file_upload, clear_old_output, save_uploaded_file, update_file_list
from download import download_callback
from model import get_model_config, MODEL_CONFIGS
from processing import process_audio, auto_ensemble_process, ensemble_audio_fn

# BASE_DIR tanımı (BASE_PATH yerine)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Arayüz oluşturma fonksiyonu
def create_interface():
    # CSS tanımı
    css = """
    /* Genel Tema */
    body {
        background: url('/content/logo.jpg') no-repeat center center fixed;
        background-size: cover;
        background-color: #2d0b0b; /* Koyu kırmızı, dublaj stüdyosuna uygun */
        min-height: 100vh;
        margin: 0;
        padding: 1rem;
        font-family: 'Poppins', sans-serif;
        color: #C0C0C0; /* Metalik gümüş metin, profesyonel görünüm */
    }

    body::after {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(45, 11, 11, 0.9); /* Daha koyu kırmızı overlay */
        z-index: -1;
    }

    /* Logo Stilleri */
    .logo-container {
        position: absolute;
        top: 1rem;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        align-items: center;
        z-index: 2000; /* Diğer öğelerden üstte, mutlaka görünür */
    }

    .logo-img {
        width: 120px;
        height: auto;
    }

    /* Başlık Stilleri */
    .header-text {
        text-align: center;
        padding: 80px 20px 20px; /* Logo için alan bırak */
        color: #ff4040; /* Kırmızı, dublaj temasına uygun */
        font-size: 2.5rem; /* Daha etkileyici ve büyük başlık */
        font-weight: 900; /* Daha kalın ve dramatik */
        text-shadow: 0 0 10px rgba(255, 64, 64, 0.5); /* Kırmızı gölge efekti */
        z-index: 1500; /* Tablerden üstte, logonun altında */
    }

    /* Metalik kırmızı parlama animasyonu */
    @keyframes metallic-red-shine {
        0% { filter: brightness(1) saturate(1) drop-shadow(0 0 5px #ff4040); }
        50% { filter: brightness(1.3) saturate(1.7) drop-shadow(0 0 15px #ff6b6b); }
        100% { filter: brightness(1) saturate(1) drop-shadow(0 0 5px #ff4040); }
    }

    /* Dublaj temalı stil */
    .dubbing-theme {
        background: linear-gradient(to bottom, #800000, #2d0b0b); /* Koyu kırmızı gradyan */
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 10px 20px rgba(255, 64, 64, 0.3); /* Kırmızı gölge */
    }

    /* Footer Stilleri (Tablerin Üstünde, Şeffaf) */
    .footer {
        text-align: center;
        padding: 10px;
        color: #ff4040; /* Kırmızı metin, dublaj temasına uygun */
        font-size: 14px;
        margin-top: 20px;
        position: relative;
        z-index: 1001; /* Tablerden üstte, logodan düşük */
    }

    /* Düğme ve Yükleme Alanı Stilleri */
    button {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        background: #800000 !important; /* Koyu kırmızı, dublaj temasına uygun */
        border: 1px solid #ff4040 !important; /* Kırmızı sınır */
        color: #C0C0C0 !important; /* Metalik gümüş metin */
        border-radius: 8px !important;
        padding: 8px 16px !important;
        position: relative;
        overflow: hidden !important;
        font-size: 0.9rem !important;
    }

    button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 10px 40px rgba(255, 64, 64, 0.7) !important; /* Daha belirgin kırmızı gölge */
        background: #ff4040 !important; /* Daha açık kırmızı hover efekti */
    }

    button::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, 
            transparent 20%, 
            rgba(192, 192, 192, 0.3) 50%, /* Metalik gümüş ton */
            transparent 80%);
        animation: button-shine 3s infinite linear;
    }

    /* Resim ve Ses Yükleme Alanı Stili */
    .compact-upload.horizontal {
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
        max-width: 400px !important;
        height: 40px !important;
        padding: 0 12px !important;
        border: 1px solid #ff4040 !important; /* Kırmızı sınır */
        background: rgba(128, 0, 0, 0.5) !important; /* Koyu kırmızı, şeffaf */
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        color: #C0C0C0 !important; /* Metalik gümüş metin */
    }

    .compact-upload.horizontal:hover {
        border-color: #ff6b6b !important; /* Daha açık kırmızı */
        background: rgba(128, 0, 0, 0.7) !important; /* Daha koyu kırmızı hover */
    }

    .compact-upload.horizontal .w-full {
        flex: 1 1 auto !important;
        min-width: 120px !important;
        margin: 0 !important;
        color: #C0C0C0 !important; /* Metalik gümüş */
    }

    .compact-upload.horizontal button {
        padding: 4px 12px !important;
        font-size: 0.75em !important;
        height: 28px !important;
        min-width: 80px !important;
        border-radius: 4px !important;
        background: #800000 !important; /* Koyu kırmızı */
        border: 1px solid #ff4040 !important; /* Kırmızı sınır */
        color: #C0C0C0 !important; /* Metalik gümüş */
    }

    .compact-upload.horizontal .text-gray-500 {
        font-size: 0.7em !important;
        color: rgba(192, 192, 192, 0.6) !important; /* Şeffaf metalik gümüş */
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        max-width: 180px !important;
    }

    /* Ekstra Dar Versiyon */
    .compact-upload.horizontal.x-narrow {
        max-width: 320px !important;
        height: 36px !important;
        padding: 0 10px !important;
        gap: 6px !important;
    }

    .compact-upload.horizontal.x-narrow button {
        padding: 3px 10px !important;
        font-size: 0.7em !important;
        height: 26px !important;
        min-width: 70px !important;
    }

    .compact-upload.horizontal.x-narrow .text-gray-500 {
        font-size: 0.65em !important;
        max-width: 140px !important;
    }

    /* Sekmeler İçin Ortak Stiller */
    .gr-tab {
        background: rgba(128, 0, 0, 0.5) !important; /* Koyu kırmızı, şeffaf */
        border-radius: 12px 12px 0 0 !important;
        margin: 0 5px !important;
        color: #C0C0C0 !important; /* Metalik gümüş */
        border: 1px solid #ff4040 !important; /* Kırmızı sınır */
        z-index: 1500; /* Logo’nun altında, diğer öğelerden üstte */
    }

    .gr-tab-selected {
        background: #800000 !important; /* Koyu kırmızı */
        box-shadow: 0 4px 12px rgba(255, 64, 64, 0.7) !important; /* Daha belirgin kırmızı gölge */
        color: #ffffff !important; /* Beyaz metin (seçili sekme için kontrast) */
        border: 1px solid #ff6b6b !important; /* Daha açık kırmızı */
    }

    /* Manuel Ensemble Özel Stilleri */
    .compact-header {
        font-size: 0.95em !important;
        margin: 0.8rem 0 0.5rem 0 !important;
        color: #C0C0C0 !important; /* Metalik gümüş metin */
    }

    .compact-grid {
        gap: 0.4rem !important;
        max-height: 50vh;
        overflow-y: auto;
        padding: 10px;
        background: rgba(128, 0, 0, 0.3) !important; /* Koyu kırmızı, şeffaf */
        border-radius: 12px;
        border: 1px solid #ff4040 !important; /* Kırmızı sınır */
    }

    .compact-dropdown {
        --padding: 8px 12px !important;
        --radius: 10px !important;
        border: 1px solid #ff4040 !important; /* Kırmızı sınır */
        background: rgba(128, 0, 0, 0.5) !important; /* Koyu kırmızı, şeffaf */
        color: #C0C0C0 !important; /* Metalik gümüş metin */
    }

    .tooltip-icon {
        font-size: 1.4em !important;
        color: #C0C0C0 !important; /* Metalik gümüş */
        cursor: help;
        margin-left: 0.5rem !important;
    }

    .log-box {
        font-family: 'Fira Code', monospace !important;
        font-size: 0.85em !important;
        background-color: rgba(128, 0, 0, 0.3) !important; /* Koyu kırmızı, şeffaf */
        border: 1px solid #ff4040 !important; /* Kırmızı sınır */
        border-radius: 8px;
        padding: 1rem !important;
        color: #C0C0C0 !important; /* Metalik gümüş metin */
    }

    /* Animasyonlar */
    @keyframes text-glow {
        0% { text-shadow: 0 0 5px rgba(192, 192, 192, 0); }
        50% { text-shadow: 0 0 15px rgba(192, 192, 192, 1); }
        100% { text-shadow: 0 0 5px rgba(192, 192, 192, 0); }
    }

    @keyframes button-shine {
        0% { transform: rotate(0deg) translateX(-50%); }
        100% { transform: rotate(360deg) translateX(-50%); }
    }

    /* Responsive Ayarlar */
    @media (max-width: 768px) {
        .compact-grid {
            max-height: 40vh;
        }

        .compact-upload.horizontal {
            max-width: 100% !important;
            width: 100% !important;
        }
        
        .compact-upload.horizontal .text-gray-500 {
            max-width: 100px !important;
        }
        
        .compact-upload.horizontal.x-narrow {
            height: 40px !important;
            padding: 0 8px !important;
        }

        .logo-container {
            width: 80px; /* Mobil cihazlarda daha küçük logo */
            top: 1rem;
            left: 50%;
            transform: translateX(-50%);
        }

        .header-text {
            padding: 60px 20px 20px; /* Mobil için daha az boşluk */
            font-size: 1.8rem; /* Mobil için biraz daha küçük başlık */
        }
    }
    """

    # Arayüz tasarımı
    with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
        with gr.Column():
            # Logo (PNG olarak, dublaj temasına uygun)
            logo_html = """
            <div class="logo-container">
                <img src="/content/gk_logo.png" alt="" class="logo-img">
            </div>
            """
            gr.HTML(logo_html)

            # Başlık (Etkileyici ve dublaj temalı)
            gr.HTML("""
            <div class="header-text">
                Gecekondu Dubbing Production
            </div>
            """)

        with gr.Tabs():
            with gr.Tab("Audio Separation", elem_id="separation_tab"):
                with gr.Row(equal_height=True):
                    # Sol Panel - Kontroller
                    with gr.Column(scale=1, min_width=380):
                        with gr.Accordion("📥 Input & Model", open=True):
                            with gr.Tabs():
                                with gr.Tab("🖥 Upload"):
                                    input_audio_file = gr.File(
                                        file_types=[".wav", ".mp3", ".m4a", ".mp4", ".mkv", ".flac"],
                                        elem_classes=["compact-upload", "horizontal", "x-narrow"],
                                        label=""
                                    )

                                with gr.Tab("📂 Path"):
                                    file_path_input = gr.Textbox(placeholder="/path/to/audio.wav")

                            with gr.Row():
                                model_category = gr.Dropdown(
                                    label="Category",
                                    choices=list(MODEL_CONFIGS.keys()),
                                    value="Vocal Models"
                                )
                                model_dropdown = gr.Dropdown(
                                    label="Model",
                                    choices=list(MODEL_CONFIGS["Vocal Models"].keys())
                                )

                        with gr.Accordion("⚙ Settings", open=False):
                            with gr.Row():
                                export_format = gr.Dropdown(
                                    label="Format",
                                    choices=['wav FLOAT', 'flac PCM_16', 'flac PCM_24'],
                                    value='wav FLOAT'
                                )
                                chunk_size = gr.Dropdown(
                                    label="Chunk Size",
                                    choices=[352800, 485100],
                                    value=352800,
                                    info="Don't change unless you have specific requirements"
                                )

                            with gr.Row():
                                overlap = gr.Slider(2, 50, step=1, label="Overlap", value=2)
                                gr.Markdown("Recommended: 2-10 (Higher values increase quality but require more VRAM)")
                                use_tta = gr.Checkbox(label="TTA Boost")

                            with gr.Row():
                                use_demud_phaseremix_inst = gr.Checkbox(label="Phase Fix")
                                gr.Markdown("Advanced phase correction for instrumental tracks")
                                extract_instrumental = gr.Checkbox(label="Instrumental")

                        with gr.Row():
                            process_btn = gr.Button("🚀 Process", variant="primary")
                            clear_old_output_btn = gr.Button("🧹 Reset", variant="secondary")
                            clear_old_output_status = gr.Textbox(label="Status", interactive=False)

                    # Sağ Panel - Sonuçlar
                    with gr.Column(scale=2, min_width=800):
                        with gr.Tabs():
                            with gr.Tab("🎧 Main"):
                                with gr.Column():
                                    original_audio = gr.Audio(label="Original", interactive=False)
                                    with gr.Row():
                                        vocals_audio = gr.Audio(label="Vocals", show_download_button=True)
                                        instrumental_audio = gr.Audio(label="Instrumental", show_download_button=True)

                            with gr.Tab("🔍 Details"):
                                with gr.Column():
                                    with gr.Row():
                                        male_audio = gr.Audio(label="Male")
                                        female_audio = gr.Audio(label="Female")
                                        speech_audio = gr.Audio(label="Speech")
                                    with gr.Row():
                                        drum_audio = gr.Audio(label="Drums")
                                        bass_audio = gr.Audio(label="Bass")
                                    with gr.Row():
                                        other_audio = gr.Audio(label="Other")
                                        effects_audio = gr.Audio(label="Effects")

                            with gr.Tab("⚙ Advanced"):
                                with gr.Column():
                                    with gr.Row():
                                        phaseremix_audio = gr.Audio(label="Phase Remix")
                                        dry_audio = gr.Audio(label="Dry")
                                    with gr.Row():
                                        music_audio = gr.Audio(label="Music")
                                        karaoke_audio = gr.Audio(label="Karaoke")
                                        bleed_audio = gr.Audio(label="Bleed")

                            with gr.Row():
                                gr.Markdown("""
                                <div style="
                                    background: rgba(245,245,220,0.15);
                                    padding: 1rem;
                                    border-radius: 8px;
                                    border-left: 3px solid #6c757d;
                                    margin: 1rem 0 1.5rem 0;
                                ">
                                    <b>🔈 Processing Tip:</b> For noisy results, use <code>bleed_suppressor_v1</code> 
                                    or <code>denoisedebleed</code> models in the <i>"Denoise & Effect Removal"</i> 
                                    category to clean the output
                                </div>
                                """)

            # Oto Ensemble Sekmesi
            with gr.Tab("Auto Ensemble"):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            auto_input_audio_file = gr.File(label="Upload file")
                            auto_file_path_input = gr.Textbox(
                                label="Or enter file path",
                                placeholder="Enter full path to audio file",
                                interactive=True
                            )

                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            with gr.Row():
                                auto_use_tta = gr.Checkbox(label="Use TTA", value=False)
                                auto_extract_instrumental = gr.Checkbox(label="Instrumental Only")

                            with gr.Row():
                                auto_overlap = gr.Slider(
                                    label="Overlap",
                                    minimum=2,
                                    maximum=50,
                                    value=2,
                                    step=1
                                )
                                auto_chunk_size = gr.Dropdown(
                                    label="Chunk Size",
                                    choices=[352800, 485100],
                                    value=352800
                                )
                                export_format2 = gr.Dropdown(
                                    label="Output Format",
                                    choices=['wav FLOAT', 'flac PCM_16', 'flac PCM_24'],
                                    value='wav FLOAT'
                                )

                        with gr.Group():
                            gr.Markdown("### 🧠 Model Selection")
                            with gr.Row():
                                auto_category_dropdown = gr.Dropdown(
                                    label="Model Category",
                                    choices=list(MODEL_CONFIGS.keys()),
                                    value="Vocal Models"
                                )

                            auto_model_dropdown = gr.Dropdown(
                                label="Select Models from Category",
                                choices=list(MODEL_CONFIGS["Vocal Models"].keys()),
                                multiselect=True,
                                max_choices=50,
                                interactive=True
                            )

                            selected_models = gr.Dropdown(
                                label="Selected Models",
                                choices=[],
                                multiselect=True,
                                interactive=False
                            )

                            with gr.Row():
                                add_btn = gr.Button("➕ Add Selected", variant="secondary")
                                clear_btn = gr.Button("🗑️ Clear All", variant="stop")

                        with gr.Group():
                            gr.Markdown("### ⚡ Ensemble Settings")
                            with gr.Row():
                                auto_ensemble_type = gr.Dropdown(
                                    label="Method",
                                    choices=['avg_wave', 'median_wave', 'min_wave', 'max_wave',
                                            'avg_fft', 'median_fft', 'min_fft', 'max_fft'],
                                    value='avg_wave'
                                )

                            gr.Markdown("**Recommendation:** avg_wave and max_fft best results")

                        auto_process_btn = gr.Button("🚀 Start Processing", variant="primary")

                    with gr.Column():
                        with gr.Tabs():
                            with gr.Tab("🔊 Original Audio"):
                                original_audio2 = gr.Audio(
                                    label="Original Audio",
                                    interactive=False,
                                    every=1,
                                    elem_id="original_audio_player"
                                )
                            with gr.Tab("🎚️ Ensemble Result"):
                                auto_output_audio = gr.Audio(
                                    label="Output Preview",
                                    show_download_button=True,
                                    interactive=False
                                )

                        auto_status = gr.Textbox(
                            label="Processing Status",
                            interactive=False,
                            placeholder="Waiting for processing...",
                            elem_classes="status-box"
                        )

                        gr.Markdown("""
                            <div style="
                                background: rgba(110, 142, 251, 0.1);
                                padding: 1.2rem;
                                border-radius: 12px;
                                border-left: 4px solid #6e8efb;
                                margin: 1rem 0;
                                backdrop-filter: blur(3px);
                                border: 1px solid rgba(255,255,255,0.2);
                            ">
                                <div style="display: flex; align-items: start; gap: 1rem;">
                                    <div style="
                                        font-size: 1.4em;
                                        color: #6e8efb;
                                        margin-top: -2px;
                                    ">⚠️</div>
                                    <div style="color: #2d3748;">
                                        <h4 style="
                                            margin: 0 0 0.8rem 0;
                                            color: #4a5568;
                                            font-weight: 600;
                                            font-size: 1.1rem;
                                        ">
                                            Model Selection Guidelines
                                        </h4>
                                        <ul style="
                                            margin: 0;
                                            padding-left: 1.2rem;
                                            color: #4a5568;
                                            line-height: 1.6;
                                        ">
                                            <li><strong>Avoid cross-category mixing:</strong> Combining vocal and instrumental models may create unwanted blends</li>
                                            <li><strong>Special model notes:</strong>
                                                <ul style="padding-left: 1.2rem; margin: 0.5rem 0;">
                                                    <li>Duality models (v1/v2) - Output both stems</li>
                                                    <li>MDX23C Separator - Hybrid results</li>
                                                </ul>
                                            </li>
                                            <li><strong>Best practice:</strong> Use 3-5 similar models from same category</li>
                                        </ul>
                                        <div style="
                                            margin-top: 1rem;
                                            padding: 0.8rem;
                                            background: rgba(167, 119, 227, 0.1);
                                            border-radius: 8px;
                                            color: #6e8efb;
                                            font-size: 0.9rem;
                                        ">
                                            💡 Pro Tip: Start with "VOCALS-MelBand-Roformer BigBeta5e" + "VOCALS-BS-Roformer_1297" combination
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """)

            # İndirme Sekmesi
            with gr.Tab("Download Sources"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🗂️ Cloud Storage")
                        drive_url_input = gr.Textbox(label="Google Drive Shareable Link")
                        drive_download_btn = gr.Button("⬇️ Download from Drive", variant="secondary")
                        drive_download_status = gr.Textbox(label="Download Status")
                        drive_download_output = gr.File(label="Downloaded File", interactive=False)

                    with gr.Column():
                        gr.Markdown("### 🌐 Direct Links")
                        direct_url_input = gr.Textbox(label="Audio File URL")
                        direct_download_btn = gr.Button("⬇️ Download from URL", variant="secondary")
                        direct_download_status = gr.Textbox(label="Download Status")
                        direct_download_output = gr.File(label="Downloaded File", interactive=False)

                    with gr.Column():
                        gr.Markdown("### 🍪 Cookie Management")
                        cookie_file = gr.File(
                            label="Upload Cookies.txt",
                            file_types=[".txt"],
                            interactive=True,
                            elem_id="cookie_upload"
                        )
                        gr.Markdown("""
                        <div style="margin-left:15px; font-size:0.95em">
                        **📌 Why Needed?**  
                        - Access age-restricted content  
                        - Download private/unlisted videos  
                        - Bypass regional restrictions  
                        - Avoid YouTube download limits  

                        **⚠️ Important Notes**  
                        - NEVER share your cookie files!  
                        - Refresh cookies when:  
                          • Getting "403 Forbidden" errors  
                          • Downloads suddenly stop  
                          • Seeing "Session expired" messages  

                        **🔄 Renewal Steps**  
                        1. Install this <a href="https://chromewebstore.google.com/detail/get-cookiestxt-clean/ahmnmhfbokciafffnknlekllgcnafnie" target="_blank">Chrome extension</a>  
                        2. Login to YouTube in Chrome  
                        3. Click extension icon → "Export"  
                        4. Upload the downloaded file here  

                        **⏳ Cookie Lifespan**  
                        - Normal sessions: 24 hours  
                        - Sensitive operations: 1 hour  
                        - Password changes: Immediate invalidation  
                        </div>
                        """)

            # Manuel Ensemble Sekmesi
            with gr.Tab("🎚️ Manuel Ensemble"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=400):
                        with gr.Accordion("📂 Input Sources", open=True):
                            with gr.Row():
                                refresh_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm")
                                ensemble_type = gr.Dropdown(
                                    label="Ensemble Algorithm",
                                    choices=['avg_wave', 'median_wave', 'min_wave', 'max_wave',
                                            'avg_fft', 'median_fft', 'min_fft', 'max_fft'],
                                    value='avg_wave'
                                )

                            # Dosya listesini belirli bir yoldan al
                            file_path = os.path.join(Path.home(), 'Music-Source-Separation', 'output')
                            initial_files = glob.glob(f"{file_path}/*.wav") + glob.glob(os.path.join(BASE_DIR, 'Music-Source-Separation-Training', 'old_output', '*.wav'))

                            gr.Markdown("### Select Audio Files")
                            file_dropdown = gr.Dropdown(
                                choices=initial_files,
                                label="Available Files",
                                multiselect=True,
                                interactive=True,
                                elem_id="file-dropdown"
                            )

                            weights_input = gr.Textbox(
                                label="Custom Weights (comma separated)",
                                placeholder="Example: 0.8, 1.2, 1.0, ...",
                                info="Leave empty for equal weights"
                            )

                    # Sağ Panel - Sonuçlar
                    with gr.Column(scale=2, min_width=800):
                        with gr.Tabs():
                            with gr.Tab("🎧 Result Preview"):
                                ensemble_output_audio = gr.Audio(
                                    label="Ensembled Output",
                                    interactive=False,
                                    show_download_button=True,
                                    elem_id="output-audio"
                                )

                            with gr.Tab("📋 Processing Log"):
                                ensemble_status = gr.Textbox(
                                    label="Processing Details",
                                    interactive=False,
                                    elem_id="log-box"
                                )

                            with gr.Row():
                                ensemble_process_btn = gr.Button(
                                    "⚡ Process Ensemble",
                                    variant="primary",
                                    size="sm",
                                    elem_id="process-btn"
                                )

        gr.HTML("""
        <div class="footer">
            Presented by Gecekondu Production
        </div>
        """)

        def update_models(category):
            return gr.Dropdown(choices=list(MODEL_CONFIGS[category].keys()))

        def add_models(new_models, existing_models):
            updated = list(set(existing_models + new_models))
            return gr.Dropdown(choices=updated, value=updated)

        def clear_models():
            return gr.Dropdown(choices=[], value=[])

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

        auto_category_dropdown.change(fn=update_models, inputs=auto_category_dropdown, outputs=auto_model_dropdown)
        add_btn.click(fn=add_models, inputs=[auto_model_dropdown, selected_models], outputs=selected_models)
        clear_btn.click(fn=clear_models, inputs=[], outputs=selected_models)

        process_btn.click(
            fn=process_audio,
            inputs=[
                input_audio_file, model_dropdown, chunk_size, overlap, export_format,
                use_tta, use_demud_phaseremix_inst, extract_instrumental, gr.State(None), gr.State(None)
            ],
            outputs=[
                vocals_audio, instrumental_audio, phaseremix_audio, drum_audio, karaoke_audio,
                bass_audio, other_audio, effects_audio, speech_audio, bleed_audio, music_audio,
                dry_audio, male_audio, female_audio
            ]
        )

        auto_process_btn.click(
            fn=auto_ensemble_process,
            inputs=[
                auto_input_audio_file, selected_models, auto_chunk_size, auto_overlap, export_format2,
                auto_use_tta, auto_extract_instrumental, auto_ensemble_type, gr.State(None)
            ],
            outputs=[auto_output_audio, auto_status]
        )

        drive_download_btn.click(
            fn=download_callback,
            inputs=[drive_url_input, gr.State('drive')],
            outputs=[drive_download_output, drive_download_status, input_audio_file, auto_input_audio_file, original_audio, original_audio2]
        )

        direct_download_btn.click(
            fn=download_callback,
            inputs=[direct_url_input, gr.State('direct'), cookie_file],
            outputs=[direct_download_output, direct_download_status, input_audio_file, auto_input_audio_file, original_audio, original_audio2]
        )

        refresh_btn.click(fn=update_file_list, outputs=file_dropdown)
        ensemble_process_btn.click(fn=ensemble_audio_fn, inputs=[file_dropdown, ensemble_type, weights_input], outputs=[ensemble_output_audio, ensemble_status])

    return demo
