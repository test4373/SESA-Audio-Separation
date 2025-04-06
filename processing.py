import os
import glob
import subprocess
import time
import gc
import shutil
import sys
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from datetime import datetime
from helpers import INPUT_DIR, OLD_OUTPUT_DIR, ENSEMBLE_DIR, AUTO_ENSEMBLE_TEMP, move_old_files, clear_directory, BASE_DIR
from model import get_model_config
import torch
import yaml
import gradio as gr
import threading
import random
import librosa
import soundfile as sf
import numpy as np
import requests
import json
import locale
import re
import psutil
import concurrent.futures
from tqdm import tqdm
from google.oauth2.credentials import Credentials
import tempfile
from urllib.parse import urlparse, quote
from clean_model import clean_model_name, shorten_filename, clean_filename
from google.colab import drive

import warnings
warnings.filterwarnings("ignore")

# BASE_DIR'i dinamik olarak güncel dizine ayarla
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # processing.py'nin bulunduğu dizin
INFERENCE_PATH = os.path.join(BASE_DIR, "inference.py")  # inference.py'nin tam yolu
OUTPUT_DIR = os.path.join(BASE_DIR, "output")  # Çıkış dizini BASE_DIR/output olarak güncellendi
AUTO_ENSEMBLE_OUTPUT = os.path.join(BASE_DIR, "ensemble_output")  # Ensemble çıkış dizini

def copy_ensemble_to_drive():
    """Copies the latest ensemble output file to Google Drive if mounted."""
    try:
        # Check if Google Drive is mounted
        if not os.path.exists('/content/drive'):
            drive.mount('/content/drive')
            status = "Google Drive mounted. Copying ensemble output..."
        else:
            status = "Google Drive already mounted. Copying ensemble output..."

        # Define source directory and find the latest file
        source_dir = AUTO_ENSEMBLE_OUTPUT  # AUTO_ENSEMBLE_OUTPUT from processing.py
        output_files = glob.glob(os.path.join(source_dir, "*.wav"))
        if not output_files:
            return "❌ No ensemble output files found."

        # En son oluşturulan dosyayı bul
        latest_file = max(output_files, key=os.path.getctime)
        filename = os.path.basename(latest_file)

        # Define destination path
        dest_dir = "/content/drive/MyDrive/SESA_Ensemble_Output"  # Customize this path as needed
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)

        # Copy the latest file to Drive
        shutil.copy2(latest_file, dest_path)

        return f"✅ Ensemble output copied to {dest_path}"
    except Exception as e:
        return f"❌ Error copying ensemble output: {str(e)}"

def copy_to_drive():
    """Copies processed files from OUTPUT_DIR to Google Drive if mounted."""
    try:
        # Check if Google Drive is mounted
        if not os.path.exists('/content/drive'):
            drive.mount('/content/drive')
            status = "Google Drive mounted. Copying files..."
        else:
            status = "Google Drive already mounted. Copying files..."

        # Define source and destination paths
        source_dir = OUTPUT_DIR  # Assuming OUTPUT_DIR is defined globally
        dest_dir = "/content/drive/MyDrive/SESA_Output"  # Customize this path as needed

        # Create destination directory if it doesn't exist
        os.makedirs(dest_dir, exist_ok=True)

        # Copy all files from OUTPUT_DIR to Drive
        for filename in os.listdir(source_dir):
            src_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dest_path)

        return f"✅ Files copied to {dest_dir}"
    except Exception as e:
        return f"❌ Error copying files: {str(e)}"  

def refresh_auto_output():
    """AUTO_ENSEMBLE_OUTPUT dizinindeki en son dosyayı bulur ve döndürür."""
    try:
        output_files = glob.glob(os.path.join(AUTO_ENSEMBLE_OUTPUT, "*.wav"))
        if not output_files:
            return None, i18n("no_output_files_found")
        
        latest_file = max(output_files, key=os.path.getctime)
        return latest_file, i18n("output_refreshed_successfully")
    except Exception as e:
        return None, i18n("error_refreshing_output").format(str(e))

def extract_model_name(full_model_string):
    """Bir dizeden temiz model adını çıkarır."""
    if not full_model_string:
        return ""
    cleaned = str(full_model_string)
    if ' - ' in cleaned:
        cleaned = cleaned.split(' - ')[0]
    emoji_prefixes = ['✅ ', '👥 ', '🗣️ ', '🏛️ ', '🔇 ', '🔉 ', '🎬 ', '🎼 ', '✅(?) ']
    for prefix in emoji_prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    return cleaned.strip()

def run_command_and_process_files(model_type, config_path, start_check_point, INPUT_DIR, OUTPUT_DIR, extract_instrumental, use_tta, demud_phaseremix_inst, clean_model, progress=gr.Progress()):
    try:
        cmd_parts = [
            "python", INFERENCE_PATH,
            "--model_type", model_type,
            "--config_path", config_path,
            "--start_check_point", start_check_point,
            "--input_folder", INPUT_DIR,
            "--store_dir", OUTPUT_DIR,
        ]
        if extract_instrumental:
            cmd_parts.append("--extract_instrumental")
        if use_tta:
            cmd_parts.append("--use_tta")
        if demud_phaseremix_inst:
            cmd_parts.append("--demud_phaseremix_inst")

        process = subprocess.Popen(
            cmd_parts,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        progress(0, desc=i18n("starting_audio_separation"), total=100)
        progress_bar = tqdm(total=100, desc=i18n("processing_audio"), unit="%")

        for line in process.stdout:
            print(line.strip())
            if "Progress:" in line:
                try:
                    percentage = float(re.search(r"Progress: (\d+\.\d+)%", line).group(1))
                    progress(percentage, desc=i18n("separating_audio").format(percentage))
                    progress_bar.n = percentage
                    progress_bar.refresh()
                except (AttributeError, ValueError) as e:
                    print(i18n("progress_parsing_error").format(e))
            elif "Processing file" in line:
                progress(0, desc=line.strip())

        for line in process.stderr:
            print(line.strip())

        process.wait()
        progress_bar.close()
        progress(100, desc=i18n("separation_complete"))

        filename_model = clean_model_name(clean_model)

        def rename_files_with_model(folder, filename_model):
            for filename in sorted(os.listdir(folder)):
                file_path = os.path.join(folder, filename)
                if not any(filename.lower().endswith(ext) for ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']):
                    continue
                base, ext = os.path.splitext(filename)
                clean_base = base.strip('_- ')
                new_filename = f"{clean_base}_{filename_model}{ext}"
                new_file_path = os.path.join(folder, new_filename)
                os.rename(file_path, new_file_path)

        rename_files_with_model(OUTPUT_DIR, filename_model)

        output_files = os.listdir(OUTPUT_DIR)

        def find_file(keyword):
            matching_files = [
                os.path.join(OUTPUT_DIR, f) for f in output_files 
                if keyword in f.lower()
            ]
            return matching_files[0] if matching_files else None

        vocal_file = find_file('vocals')
        instrumental_file = find_file('instrumental')
        phaseremix_file = find_file('phaseremix')
        drum_file = find_file('drum')
        bass_file = find_file('bass')
        other_file = find_file('other')
        effects_file = find_file('effects')
        speech_file = find_file('speech')
        music_file = find_file('music')
        dry_file = find_file('dry')
        male_file = find_file('male')
        female_file = find_file('female')
        bleed_file = find_file('bleed')
        karaoke_file = find_file('karaoke')

        return (
            vocal_file or None,
            instrumental_file or None,
            phaseremix_file or None,
            drum_file or None,
            bass_file or None,
            other_file or None,
            effects_file or None,
            speech_file or None,
            music_file or None,
            dry_file or None,
            male_file or None,
            female_file or None,
            bleed_file or None,
            karaoke_file or None
        )

    except Exception as e:
        print(i18n("error_occurred").format(e))
        return (None,) * 14

    finally:
        progress(100, desc=i18n("separation_process_completed"))

def process_audio(input_audio_file, model, chunk_size, overlap, export_format, use_tta, demud_phaseremix_inst, extract_instrumental, clean_model, progress=gr.Progress(track_tqdm=True), *args, **kwargs):
    """Belirtilen modeli kullanarak sesi işler ve ayrılmış stem'leri ilerleme ile birlikte döndürür."""
    try:
        if input_audio_file is not None:
            audio_path = input_audio_file.name
        else:
            existing_files = os.listdir(INPUT_DIR)
            if existing_files:
                audio_path = os.path.join(INPUT_DIR, existing_files[0])
            else:
                yield (
                    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                    i18n("no_audio_file_error"),
                    """
                    <div id="custom-progress" style="margin-top: 10px;">
                        <div style="font-size: 1rem; color: #C0C0C0; margin-bottom: 5px;" id="progress-label">{}</div>
                        <div style="width: 100%; background-color: #444; border-radius: 5px; overflow: hidden;">
                            <div id="progress-bar" style="width: 0%; height: 20px; background-color: #6e8efb; transition: width 0.3s;"></div>
                        </div>
                    </div>
                    """.format(i18n("no_input_progress_label"))
                )
                return

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(OLD_OUTPUT_DIR, exist_ok=True)
        move_old_files(OUTPUT_DIR)

        clean_model_name_full = extract_model_name(model)
        print(i18n("processing_audio_print").format(audio_path, clean_model_name_full))

        progress_html = """
        <div id="custom-progress" style="margin-top: 10px;">
            <div style="font-size: 1rem; color: #C0C0C0; margin-bottom: 5px;" id="progress-label">{}</div>
            <div style="width: 100%; background-color: #444; border-radius: 5px; overflow: hidden;">
                <div id="progress-bar" style="width: 0%; height: 20px; background-color: #6e8efb; transition: width 0.3s;"></div>
            </div>
        </div>
        """.format(i18n("starting_audio_separation_progress_label"))
        yield (
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            i18n("starting_audio_separation"),
            progress_html
        )

        model_type, config_path, start_check_point = get_model_config(clean_model_name_full, chunk_size, overlap)

        outputs = run_command_and_process_files(
            model_type=model_type,
            config_path=config_path,
            start_check_point=start_check_point,
            INPUT_DIR=INPUT_DIR,
            OUTPUT_DIR=OUTPUT_DIR,
            extract_instrumental=extract_instrumental,
            use_tta=use_tta,
            demud_phaseremix_inst=demud_phaseremix_inst,
            clean_model=clean_model_name_full,
            progress=progress
        )

        for i in range(10, 91, 10):
            progress_html = """
            <div id="custom-progress" style="margin-top: 10px;">
                <div style="font-size: 1rem; color: #C0C0C0; margin-bottom: 5px;" id="progress-label">{}</div>
                <div style="width: 100%; background-color: #444; border-radius: 5px; overflow: hidden;">
                    <div id="progress-bar" style="width: {}%; height: 20px; background-color: #6e8efb; transition: width 0.3s;"></div>
                </div>
            </div>
            """.format(i18n("separating_audio_progress_label").format(i), i)
            yield (
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                i18n("separating_audio").format(i),
                progress_html
            )
            time.sleep(0.1)

        progress_html = """
        <div id="custom-progress" style="margin-top: 10px;">
            <div style="font-size: 1rem; color: #C0C0C0; margin-bottom: 5px;" id="progress-label">{}</div>
            <div style="width: 100%; background-color: #444; border-radius: 5px; overflow: hidden;">
                <div id="progress-bar" style="width: 100%; height: 20px; background-color: #6e8efb; transition: width 0.3s;"></div>
            </div>
        </div>
        """.format(i18n("audio_processing_completed_progress_label"))
        yield (
            outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6],
            outputs[7], outputs[8], outputs[9], outputs[10], outputs[11], outputs[12], outputs[13],
            i18n("audio_processing_completed"),
            progress_html
        )

    except Exception as e:
        progress_html = """
        <div id="custom-progress" style="margin-top: 10px;">
            <div style="font-size: 1rem; color: #C0C0C0; margin-bottom: 5px;" id="progress-label">{}</div>
            <div style="width: 100%; background-color: #444; border-radius: 5px; overflow: hidden;">
                <div id="progress-bar" style="width: 0%; height: 20px; background-color: #6e8efb; transition: width 0.3s;"></div>
            </div>
        </div>
        """.format(i18n("error_occurred_progress_label"))
        yield (
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            i18n("error").format(str(e)),
            progress_html
        )

def ensemble_audio_fn(files, method, weights, progress=gr.Progress()):
    try:
        if len(files) < 2:
            return None, i18n("minimum_files_required")
        
        valid_files = [f for f in files if os.path.exists(f)]
        if len(valid_files) < 2:
            return None, i18n("valid_files_not_found")
        
        output_dir = os.path.join(BASE_DIR, "ensembles")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{output_dir}/ensemble_{timestamp}.wav"
        
        ensemble_args = [
            "--files", *valid_files,
            "--type", method.lower().replace(' ', '_'),
            "--output", output_path
        ]
        
        if weights and weights.strip():
            weights_list = [str(w) for w in map(float, weights.split(','))]
            ensemble_args += ["--weights", *weights_list]
        
        progress(0, desc=i18n("starting_ensemble_process"), total=100)
        result = subprocess.run(
            ["python", "ensemble.py"] + ensemble_args,
            capture_output=True,
            text=True
        )
        
        start_time = time.time()
        total_estimated_time = 10.0
        for i in np.arange(0.1, 100.1, 0.1):
            elapsed_time = time.time() - start_time
            progress_value = min(i, (elapsed_time / total_estimated_time) * 100)
            time.sleep(0.001)
            progress(progress_value, desc=i18n("ensembling_progress").format(progress_value))
        
        progress(100, desc=i18n("finalizing_ensemble_output"))
        log = i18n("success_log").format(result.stdout) if not result.stderr else i18n("error_log").format(result.stderr)
        return output_path, log

    except Exception as e:
        return None, i18n("critical_error").format(str(e))
    finally:
        progress(100, desc=i18n("ensemble_process_completed"))

def auto_ensemble_process(input_audio_file, selected_models, chunk_size, overlap, export_format, use_tta, extract_instrumental, ensemble_type, _state, progress=gr.Progress(track_tqdm=True), *args, **kwargs):
    """Birden fazla modelle sesi işler ve ensemble işlemini ilerleme ile gerçekleştirir."""
    try:
        if not selected_models or len(selected_models) < 1:
            return None, i18n("no_models_selected"), "<div></div>"

        if input_audio_file is None:
            existing_files = os.listdir(INPUT_DIR)
            if not existing_files:
                return None, i18n("no_input_audio_provided"), "<div></div>"
            audio_path = os.path.join(INPUT_DIR, existing_files[0])
        else:
            audio_path = input_audio_file.name

        auto_ensemble_temp = os.path.join(BASE_DIR, "auto_ensemble_temp")
        os.makedirs(auto_ensemble_temp, exist_ok=True)
        os.makedirs(AUTO_ENSEMBLE_OUTPUT, exist_ok=True)
        clear_directory(auto_ensemble_temp)

        all_outputs = []
        total_models = len(selected_models)
        model_progress_per_step = 90 / total_models

        for i, model in enumerate(selected_models):
            clean_model = extract_model_name(model)
            model_output_dir = os.path.join(auto_ensemble_temp, clean_model)
            os.makedirs(model_output_dir, exist_ok=True)

            current_progress = i * model_progress_per_step
            progress_html = """
            <div id="custom-progress" style="margin-top: 10px;">
                <div style="font-size: 1rem; color: #C0C0C0; margin-bottom: 5px;" id="progress-label">{}</div>
                <div style="width: 100%; background-color: #444; border-radius: 5px; overflow: hidden;">
                    <div id="progress-bar" style="width: {}%; height: 20px; background-color: #6e8efb; transition: width 0.3s;"></div>
                </div>
            </div>
            """.format(i18n("loading_model_progress_label").format(i+1, total_models, clean_model, current_progress), current_progress)
            yield None, i18n("loading_model").format(i+1, total_models, clean_model), progress_html

            model_type, config_path, start_check_point = get_model_config(clean_model, chunk_size, overlap)

            cmd = [
                "python", INFERENCE_PATH,
                "--model_type", model_type,
                "--config_path", config_path,
                "--start_check_point", start_check_point,
                "--input_folder", INPUT_DIR,
                "--store_dir", model_output_dir,
            ]
            if use_tta:
                cmd.append("--use_tta")
            if extract_instrumental:
                cmd.append("--extract_instrumental")

            print(i18n("running_command").format(' '.join(cmd)))
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                print(result.stdout)
                if result.returncode != 0:
                    print(i18n("error").format(result.stderr))
                    return None, i18n("model_failed").format(model, result.stderr), "<div></div>"
            except Exception as e:
                return None, i18n("critical_error_with_model").format(model, str(e)), "<div></div>"

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            current_progress = (i + 1) * model_progress_per_step
            progress_html = """
            <div id="custom-progress" style="margin-top: 10px;">
                <div style="font-size: 1rem; color: #C0C0C0; margin-bottom: 5px;" id="progress-label">{}</div>
                <div style="width: 100%; background-color: #444; border-radius: 5px; overflow: hidden;">
                    <div id="progress-bar" style="width: {}%; height: 20px; background-color: #6e8efb; transition: width 0.3s;"></div>
                </div>
            </div>
            """.format(i18n("completed_model_progress_label").format(i+1, total_models, clean_model, current_progress), current_progress)
            yield None, i18n("completed_model").format(i+1, total_models, clean_model), progress_html

            model_outputs = glob.glob(os.path.join(model_output_dir, "*.wav"))
            if not model_outputs:
                raise FileNotFoundError(i18n("model_output_failed").format(model))
            all_outputs.extend(model_outputs)

        progress_html = """
        <div id="custom-progress" style="margin-top: 10px;">
            <div style="font-size: 1rem; color: #C0C0C0; margin-bottom: 5px;" id="progress-label">{}</div>
            <div style="width: 100%; background-color: #444; border-radius: 5px; overflow: hidden;">
                <div id="progress-bar" style="width: 90%; height: 20px; background-color: #6e8efb; transition: width 0.3s;"></div>
            </div>
        </div>
        """.format(i18n("waiting_for_files_progress_label"))
        yield None, i18n("waiting_for_files"), progress_html

        def wait_for_files(files, timeout=300):
            start = time.time()
            while time.time() - start < timeout:
                missing = [f for f in files if not os.path.exists(f)]
                if not missing:
                    return True
                time.sleep(5)
            raise TimeoutError(i18n("missing_files_error").format(missing[:3]))

        wait_for_files(all_outputs)

        progress_html = """
        <div id="custom-progress" style="margin-top: 10px;">
            <div style="font-size: 1rem; color: #C0C0C0; margin-bottom: 5px;" id="progress-label">{}</div>
            <div style="width: 100%; background-color: #444; border-radius: 5px; overflow: hidden;">
                <div id="progress-bar" style="width: 92%; height: 20px; background-color: #6e8efb; transition: width 0.3s;"></div>
            </div>
        </div>
        """.format(i18n("performing_ensemble_progress_label"))
        yield None, i18n("performing_ensemble"), progress_html

        quoted_files = [f'"{f}"' for f in all_outputs]
        timestamp = str(int(time.time()))
        output_path = os.path.join(AUTO_ENSEMBLE_OUTPUT, f"ensemble_{timestamp}.wav")
        
        ensemble_cmd = [
            "python", "ensemble.py",
            "--files", *quoted_files,
            "--type", ensemble_type,
            "--output", f'"{output_path}"'
        ]

        print(i18n("memory_usage_before_ensemble").format(psutil.virtual_memory().percent))
        result = subprocess.run(
            " ".join(ensemble_cmd),
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        print(i18n("memory_usage_after_ensemble").format(psutil.virtual_memory().percent))

        progress_html = """
        <div id="custom-progress" style="margin-top: 10px;">
            <div style="font-size: 1rem; color: #C0C0C0; margin-bottom: 5px;" id="progress-label">{}</div>
            <div style="width: 100%; background-color: #444; border-radius: 5px; overflow: hidden;">
                <div id="progress-bar" style="width: 98%; height: 20px; background-color: #6e8efb; transition: width 0.3s;"></div>
            </div>
        </div>
        """.format(i18n("finalizing_ensemble_output_progress_label"))
        yield None, i18n("finalizing_ensemble_output"), progress_html

        if not os.path.exists(output_path):
            raise RuntimeError(i18n("ensemble_file_creation_failed").format(output_path))
        
        progress_html = """
        <div id="custom-progress" style="margin-top: 10px;">
            <div style="font-size: 1rem; color: #C0C0C0; margin-bottom: 5px;" id="progress-label">{}</div>
            <div style="width: 100%; background-color: #444; border-radius: 5px; overflow: hidden;">
                <div id="progress-bar" style="width: 100%; height: 20px; background-color: #6e8efb; transition: width 0.3s;"></div>
            </div>
        </div>
        """.format(i18n("ensemble_completed_progress_label"))
        yield output_path, i18n("success_output_created"), progress_html

    except Exception as e:
        progress_html = """
        <div id="custom-progress" style="margin-top: 10px;">
            <div style="font-size: 1rem; color: #C0C0C0; margin-bottom: 5px;" id="progress-label">{}</div>
            <div style="width: 100%; background-color: #444; border-radius: 5px; overflow: hidden;">
                <div id="progress-bar" style="width: 0%; height: 20px; background-color: #6e8efb; transition: width 0.3s;"></div>
            </div>
        </div>
        """.format(i18n("error_occurred_progress_label"))
        yield None, i18n("error").format(str(e)), progress_html
    finally:
        shutil.rmtree(auto_ensemble_temp, ignore_errors=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
