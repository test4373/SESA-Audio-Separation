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
from helpers import INPUT_DIR, OLD_OUTPUT_DIR, ENSEMBLE_DIR, AUTO_ENSEMBLE_TEMP, move_old_files, clear_directory, BASE_DIR, clean_model, extract_model_name_from_checkpoint
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
from google.colab import drive

import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INFERENCE_PATH = os.path.join(BASE_DIR, "inference.py")
ENSEMBLE_PATH = os.path.join(BASE_DIR, "ensemble.py")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
AUTO_ENSEMBLE_OUTPUT = os.path.join(BASE_DIR, "ensemble_output")

def sanitize_filename(filename):
    base, ext = os.path.splitext(filename)
    base = re.sub(r'\.+', '_', base)
    base = re.sub(r'[#<>:"/\\|?*]', '_', base)
    base = re.sub(r'\s+', '_', base)
    base = re.sub(r'_+', '_', base)
    base = base.strip('_')
    return f"{base}{ext}"

def copy_ensemble_to_drive():
    try:
        if not os.path.exists('/content/drive'):
            drive.mount('/content/drive')
            status = i18n("drive_mounted_copying_ensemble")
        else:
            status = i18n("drive_already_mounted_copying_ensemble")

        source_dir = AUTO_ENSEMBLE_OUTPUT
        output_files = glob.glob(os.path.join(source_dir, "*.wav"))
        if not output_files:
            return i18n("no_ensemble_output_files_found")

        latest_file = max(output_files, key=os.path.getctime)
        filename = os.path.basename(latest_file)

        dest_dir = "/content/drive/MyDrive/SESA_Ensemble_Output"
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)

        shutil.copy2(latest_file, dest_path)
        return i18n("ensemble_output_copied").format(dest_path)
    except Exception as e:
        return i18n("error_copying_ensemble_output").format(str(e))

def copy_to_drive():
    try:
        if not os.path.exists('/content/drive'):
            drive.mount('/content/drive')
            status = i18n("drive_mounted_copying_files")
        else:
            status = i18n("drive_already_mounted_copying_files")

        source_dir = OUTPUT_DIR
        dest_dir = "/content/drive/MyDrive/SESA_Output"
        os.makedirs(dest_dir, exist_ok=True)

        for filename in os.listdir(source_dir):
            src_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dest_path)

        return i18n("files_copied_to_drive").format(dest_dir)
    except Exception as e:
        return i18n("error_copying_files").format(str(e))

def refresh_auto_output():
    try:
        output_files = glob.glob(os.path.join(AUTO_ENSEMBLE_OUTPUT, "*.wav"))
        if not output_files:
            return None, i18n("no_output_files_found")
        
        latest_file = max(output_files, key=os.path.getctime)
        return latest_file, i18n("output_refreshed_successfully")
    except Exception as e:
        return None, i18n("error_refreshing_output").format(str(e))

def clamp_percentage(value):
    """Helper function to clamp percentage values to the 0-100 range."""
    try:
        return min(max(float(value), 0), 100)
    except (ValueError, TypeError):
        print(f"Warning: Invalid percentage value {value}, defaulting to 0")
        return 0

def update_progress_html(progress_label, progress_percent):
    """Helper function to generate progress HTML with clamped percentage."""
    progress_percent = clamp_percentage(progress_percent)
    if progress_percent > 100:
        print(f"Warning: Progress percent {progress_percent} exceeds 100, clamping to 100")
    return f"""
    <div id="custom-progress" style="margin-top: 10px;">
        <div style="font-size: 1rem; color: #C0C0C0; margin-bottom: 5px;" id="progress-label">{progress_label}</div>
        <div style="width: 100%; background-color: #444; border-radius: 5px; overflow: hidden;">
            <div id="progress-bar" style="width: {progress_percent}%; height: 20px; background-color: #6e8efb; transition: width 0.3s; max-width: 100%;"></div>
        </div>
    </div>
    """

def extract_model_name_from_checkpoint(checkpoint_path):
    if not checkpoint_path:
        return "Unknown"
    base_name = os.path.basename(checkpoint_path)
    model_name = os.path.splitext(base_name)[0]
    print(f"Original checkpoint path: {checkpoint_path}, extracted model_name: {model_name}")
    return model_name.strip()

def run_command_and_process_files(
    model_type,
    config_path,
    start_check_point,
    INPUT_DIR,
    OUTPUT_DIR,
    extract_instrumental,
    use_tta,
    demud_phaseremix_inst,
    progress=None,
    use_apollo=True,
    apollo_normal_model="Apollo Universal Model",
    inference_chunk_size=352800,
    inference_overlap=2,
    apollo_chunk_size=19,
    apollo_overlap=2,
    apollo_method="normal_method",
    apollo_midside_model=None,
    output_format="wav"
):
    """
    Run inference.py with specified parameters and process output files.
    Args:
        model_type (str): Type of model (e.g., 'mel_band_roformer').
        config_path (str): Path to model config file.
        start_check_point (str): Path to model checkpoint.
        INPUT_DIR (str): Input folder path.
        OUTPUT_DIR (str): Output folder path.
        extract_instrumental (bool): Whether to extract instrumental.
        use_tta (bool): Whether to use test-time augmentation.
        demud_phaseremix_inst (bool): Whether to use demud phaseremix.
        progress: Gradio progress object or None.
        use_apollo (bool): Whether to use Apollo enhancement.
        apollo_normal_model (str): Apollo model for normal method.
        inference_chunk_size (int): Chunk size for inference.py (logged, not passed to command).
        inference_overlap (int): Overlap percentage for inference.py (logged, not passed to command).
        apollo_chunk_size (int): Chunk size for Apollo processing.
        apollo_overlap (int): Overlap percentage for Apollo processing.
        apollo_method (str): Apollo processing method.
        apollo_midside_model (str): Apollo model for mid-side method.
        output_format (str): Output audio format.
    Returns:
        tuple: Paths to output audio files (vocals, instrumental, etc.) or None.
    """
    try:
        print(f"run_command_and_process_files: model_type={model_type}, config_path={config_path}, start_check_point={start_check_point}, inference_chunk_size={inference_chunk_size}, inference_overlap={inference_overlap}, apollo_chunk_size={apollo_chunk_size}, apollo_overlap={apollo_overlap}, progress_type={type(progress)}")
        if not config_path:
            raise ValueError(f"Configuration path is empty for model_type: {model_type}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        if not start_check_point or not os.path.exists(start_check_point):
            raise FileNotFoundError(f"Checkpoint file not found: {start_check_point}")

        # Validate inference parameters (for logging only)
        try:
            inference_chunk_size = int(inference_chunk_size)
            inference_overlap = int(inference_overlap)
        except (TypeError, ValueError) as e:
            print(f"Invalid inference_chunk_size or inference_overlap: {e}. Defaulting to inference_chunk_size=352800, inference_overlap=2")
            inference_chunk_size = 352800
            inference_overlap = 2

        # Validate Apollo parameters
        try:
            apollo_chunk_size = int(apollo_chunk_size)
            apollo_overlap = int(apollo_overlap)
        except (TypeError, ValueError) as e:
            print(f"Invalid apollo_chunk_size or apollo_overlap: {e}. Defaulting to apollo_chunk_size=19, apollo_overlap=2")
            apollo_chunk_size = 19
            apollo_overlap = 2

        cmd_parts = [
            "python", INFERENCE_PATH,
            "--model_type", model_type,
            "--config_path", config_path,
            "--start_check_point", start_check_point,
            "--input_folder", INPUT_DIR,
            "--store_dir", OUTPUT_DIR
        ]
        if extract_instrumental:
            cmd_parts.append("--extract_instrumental")
        if use_tta:
            cmd_parts.append("--use_tta")
        if demud_phaseremix_inst:
            cmd_parts.append("--demud_phaseremix_inst")

        print(f"Running command: {' '.join(cmd_parts)}")
        process = subprocess.run(
            cmd_parts,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=True
        )

        # Log subprocess output
        print(f"Subprocess stdout: {process.stdout}")
        if process.stderr:
            print(f"Subprocess stderr: {process.stderr}")

        # Initialize progress for separation phase (0-80%) if progress is callable
        if progress is not None and callable(getattr(progress, '__call__', None)):
            progress(0, desc=i18n("starting_audio_separation"), total=100)
        else:
            print("Progress not callable or None, skipping progress update")

        # Check if output files were generated
        filename_model = extract_model_name_from_checkpoint(start_check_point)
        output_files = os.listdir(OUTPUT_DIR)
        if not output_files:
            raise FileNotFoundError("No output files generated in OUTPUT_DIR")

        def rename_files_with_model(folder, filename_model):
            for filename in sorted(os.listdir(folder)):
                file_path = os.path.join(folder, filename)
                if not any(filename.lower().endswith(ext) for ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']):
                    continue
                base, ext = os.path.splitext(filename)
                detected_type = None
                for type_key in ['vocals', 'instrumental', 'phaseremix', 'drum', 'bass', 'other', 'effects', 'speech', 'music', 'dry', 'male', 'female', 'bleed', 'karaoke']:
                    if type_key in base.lower():
                        detected_type = type_key
                        break
                type_suffix = detected_type.capitalize() if detected_type else "Processed"
                clean_base = sanitize_filename(base.split('_')[0]).rsplit('.', 1)[0]
                new_filename = f"{clean_base}_{type_suffix}_{filename_model}{ext}"
                new_file_path = os.path.join(folder, new_filename)
                try:
                    os.rename(file_path, new_file_path)
                except Exception as e:
                    print(f"Failed to rename {file_path} to {new_file_path}: {str(e)}")

        rename_files_with_model(OUTPUT_DIR, filename_model)

        output_files = os.listdir(OUTPUT_DIR)
        if not output_files:
            raise FileNotFoundError("No output files generated after renaming")

        def find_file(keyword):
            matching_files = [
                os.path.join(OUTPUT_DIR, f) for f in output_files 
                if keyword in f.lower()
            ]
            return matching_files[0] if matching_files else None

        output_list = [
            find_file('vocals'), find_file('instrumental'), find_file('phaseremix'),
            find_file('drum'), find_file('bass'), find_file('other'), find_file('effects'),
            find_file('speech'), find_file('music'), find_file('dry'), find_file('male'),
            find_file('female'), find_file('bleed'), find_file('karaoke')
        ]

        normalized_outputs = []
        for output_file in output_list:
            if output_file and os.path.exists(output_file):
                normalized_file = os.path.join(OUTPUT_DIR, f"{sanitize_filename(os.path.splitext(os.path.basename(output_file))[0])}.{output_format}")
                if output_file.endswith(f".{output_format}") and output_file != normalized_file:
                    shutil.copy(output_file, normalized_file)
                elif output_file != normalized_file:
                    audio, sr = librosa.load(output_file, sr=None, mono=False)
                    sf.write(normalized_file, audio.T if audio.ndim > 1 else audio, sr)
                else:
                    normalized_file = output_file
                normalized_outputs.append(normalized_file)
            else:
                normalized_outputs.append(output_file)

        # Only process Apollo if use_apollo is True
        if use_apollo:
            apollo_script = "/content/Apollo/inference.py"

            print(f"Apollo parameters - chunk_size: {apollo_chunk_size}, overlap: {apollo_overlap}, method: {apollo_method}, normal_model: {apollo_normal_model}, midside_model: {apollo_midside_model}")

            if apollo_method == "mid_side_method":
                if apollo_midside_model == "MP3 Enhancer":
                    ckpt = "/content/Apollo/model/pytorch_model.bin"
                    config = "/content/Apollo/configs/apollo.yaml"
                elif apollo_midside_model == "Lew Vocal Enhancer":
                    ckpt = "/content/Apollo/model/apollo_model.ckpt"
                    config = "/content/Apollo/configs/apollo.yaml"
                elif apollo_midside_model == "Lew Vocal Enhancer v2 (beta)":
                    ckpt = "/content/Apollo/model/apollo_model_v2.ckpt"
                    config = "/content/Apollo/configs/config_apollo_vocal.yaml"
                else:
                    ckpt = "/content/Apollo/model/apollo_universal_model.ckpt"
                    config = "/content/Apollo/configs/config_apollo.yaml"
            else:
                if apollo_normal_model == "MP3 Enhancer":
                    ckpt = "/content/Apollo/model/pytorch_model.bin"
                    config = "/content/Apollo/configs/apollo.yaml"
                elif apollo_normal_model == "Lew Vocal Enhancer":
                    ckpt = "/content/Apollo/model/apollo_model.ckpt"
                    config = "/content/Apollo/configs/apollo.yaml"
                elif apollo_normal_model == "Lew Vocal Enhancer v2 (beta)":
                    ckpt = "/content/Apollo/model/apollo_model_v2.ckpt"
                    config = "/content/Apollo/configs/config_apollo_vocal.yaml"
                else:
                    ckpt = "/content/Apollo/model/apollo_universal_model.ckpt"
                    config = "/content/Apollo/configs/config_apollo.yaml"

            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"Apollo checkpoint file not found: {ckpt}")
            if not os.path.exists(config):
                raise FileNotFoundError(f"Apollo config file not found: {config}")

            enhanced_files = []
            total_files = len([f for f in normalized_outputs if f and os.path.exists(f)])
            progress_per_file = 20 / total_files if total_files > 0 else 20  # 80-100% for Apollo

            for idx, output_file in enumerate(normalized_outputs):
                if output_file and os.path.exists(output_file):
                    original_file_name = sanitize_filename(os.path.splitext(os.path.basename(output_file))[0])
                    enhancement_suffix = "_Mid_Side_Enhanced" if apollo_method == "mid_side_method" else "_Enhanced"
                    enhanced_output = os.path.join(OUTPUT_DIR, f"{original_file_name}{enhancement_suffix}.{output_format}")

                    try:
                        # Update progress for Apollo processing
                        if progress is not None and callable(getattr(progress, '__call__', None)):
                            current_progress = 80 + (idx * progress_per_file)
                            current_progress = clamp_percentage(current_progress)
                            progress(current_progress, desc=f"Enhancing with Apollo... ({idx+1}/{total_files})")
                        else:
                            print(f"Progress not callable or None, skipping Apollo progress update for file {idx+1}/{total_files}")

                        if apollo_method == "mid_side_method":
                            audio, sr = librosa.load(output_file, mono=False, sr=None)
                            if audio.ndim == 1:
                                audio = np.array([audio, audio])

                            mid = (audio[0] + audio[1]) * 0.5
                            side = (audio[0] - audio[1]) * 0.5

                            mid_file = os.path.join(OUTPUT_DIR, f"{original_file_name}_mid_temp.wav")
                            side_file = os.path.join(OUTPUT_DIR, f"{original_file_name}_side_temp.wav")
                            sf.write(mid_file, mid, sr)
                            sf.write(side_file, side, sr)

                            mid_output = os.path.join(OUTPUT_DIR, f"{original_file_name}_mid_enhanced.{output_format}")
                            command_mid = [
                                "python", apollo_script,
                                "--in_wav", mid_file,
                                "--out_wav", mid_output,
                                "--chunk_size", str(int(apollo_chunk_size)),
                                "--overlap", str(int(apollo_overlap)),
                                "--ckpt", ckpt,
                                "--config", config
                            ]
                            print(f"Running Apollo Mid command: {' '.join(command_mid)}")
                            result_mid = subprocess.run(command_mid, capture_output=True, text=True)
                            if result_mid.returncode != 0:
                                print(f"Apollo Mid processing failed: {result_mid.stderr}")
                                enhanced_files.append(output_file)
                                continue

                            side_output = os.path.join(OUTPUT_DIR, f"{original_file_name}_side_enhanced.{output_format}")
                            command_side = [
                                "python", apollo_script,
                                "--in_wav", side_file,
                                "--out_wav", side_output,
                                "--chunk_size", str(int(apollo_chunk_size)),
                                "--overlap", str(int(apollo_overlap)),
                                "--ckpt", ckpt,
                                "--config", config
                            ]
                            print(f"Running Apollo Side command: {' '.join(command_side)}")
                            result_side = subprocess.run(command_side, capture_output=True, text=True)
                            if result_side.returncode != 0:
                                print(f"Apollo Side processing failed: {result_side.stderr}")
                                enhanced_files.append(output_file)
                                continue

                            if not (os.path.exists(mid_output) and os.path.exists(side_output)):
                                print(f"Apollo outputs missing: mid={mid_output}, side={side_output}")
                                enhanced_files.append(output_file)
                                continue

                            mid_audio, _ = librosa.load(mid_output, sr=sr, mono=True)
                            side_audio, _ = librosa.load(side_output, sr=sr, mono=True)
                            left = mid_audio + side_audio
                            right = mid_audio - side_audio
                            combined = np.array([left, right])

                            os.makedirs(os.path.dirname(enhanced_output), exist_ok=True)
                            sf.write(enhanced_output, combined.T, sr)

                            temp_files = [mid_file, side_file, mid_output, side_output]
                            for temp_file in temp_files:
                                try:
                                    if os.path.exists(temp_file):
                                        os.remove(temp_file)
                                except Exception as e:
                                    print(f"Failed to remove temporary file {temp_file}: {str(e)}")

                            enhanced_files.append(enhanced_output)
                        else:
                            command = [
                                "python", apollo_script,
                                "--in_wav", output_file,
                                "--out_wav", enhanced_output,
                                "--chunk_size", str(int(apollo_chunk_size)),
                                "--overlap", str(int(apollo_overlap)),
                                "--ckpt", ckpt,
                                "--config", config
                            ]
                            print(f"Running Apollo Normal command: {' '.join(command)}")
                            apollo_process = subprocess.run(
                                command,
                                capture_output=True,
                                text=True
                            )
                            if apollo_process.returncode != 0:
                                print(f"Apollo failed for {output_file}: {apollo_process.stderr}")
                                enhanced_files.append(output_file)
                                continue

                            if not os.path.exists(enhanced_output):
                                print(f"Apollo output missing: {enhanced_output}")
                                enhanced_files.append(output_file)
                                continue

                            enhanced_files.append(enhanced_output)

                        # Update progress after processing each file
                        if progress is not None and callable(getattr(progress, '__call__', None)):
                            current_progress = 80 + ((idx + 1) * progress_per_file)
                            current_progress = clamp_percentage(current_progress)
                            progress(current_progress, desc=f"Enhancing with Apollo... ({idx+1}/{total_files})")

                    except Exception as e:
                        print(f"Error during Apollo processing for {output_file}: {str(e)}")
                        enhanced_files.append(output_file)
                        continue
                else:
                    enhanced_files.append(output_file)

            # Final progress update
            if progress is not None and callable(getattr(progress, '__call__', None)):
                progress(100, desc=i18n("apollo_enhancement_complete"))
            return tuple(enhanced_files)

        # If use_apollo is False, return the normalized outputs without Apollo processing
        if progress is not None and callable(getattr(progress, '__call__', None)):
            progress(100, desc=i18n("separation_complete"))
        return tuple(normalized_outputs)

    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with code {e.returncode}: {e.stderr}")
        return (None,) * 14
    except Exception as e:
        print(f"Error in run_command_and_process_files: {str(e)}")
        import traceback
        traceback.print_exc()
        return (None,) * 14
def process_audio(input_audio_file, model, chunk_size, overlap, export_format, use_tta, demud_phaseremix_inst, extract_instrumental, use_apollo, apollo_chunk_size, apollo_overlap, apollo_method, apollo_normal_model, apollo_midside_model, progress=gr.Progress(track_tqdm=True), *args, **kwargs):
    try:
        if input_audio_file is not None:
            audio_path = input_audio_file.name
        else:
            return (
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                i18n("no_audio_file_error"),
                update_progress_html(i18n("no_input_progress_label"), 0)
            )

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(OLD_OUTPUT_DIR, exist_ok=True)
        move_old_files(OUTPUT_DIR)

        print(f"process_audio: model parameter received: {model}")
        # Clean the model name to remove ⭐ and other unwanted characters
        clean_model_name = clean_model(model) if not model.startswith("/") else extract_model_name_from_checkpoint(model)
        print(f"Processing audio from: {audio_path} using model: {clean_model_name}")

        print(f"Raw UI inputs - chunk_size: {chunk_size}, overlap: {overlap}, apollo_chunk_size: {apollo_chunk_size}, apollo_overlap: {apollo_overlap}, apollo_method: {apollo_method}")

        # Validate inference parameters (for logging only)
        try:
            inference_chunk_size = int(chunk_size)
        except (TypeError, ValueError):
            print(f"Invalid chunk_size: {chunk_size}. Defaulting to 352800.")
            inference_chunk_size = 352800

        try:
            inference_overlap = int(overlap)
        except (TypeError, ValueError):
            print(f"Invalid overlap: {overlap}. Defaulting to 2.")
            inference_overlap = 2

        # Validate Apollo parameters
        try:
            apollo_chunk_size = int(apollo_chunk_size)
        except (TypeError, ValueError):
            print(f"Invalid apollo_chunk_size: {apollo_chunk_size}. Defaulting to 19.")
            apollo_chunk_size = 19

        try:
            apollo_overlap = int(apollo_overlap)
        except (TypeError, ValueError):
            print(f"Invalid apollo_overlap: {apollo_overlap}. Defaulting to 2.")
            apollo_overlap = 2

        # Map apollo_method to backend values
        if apollo_method in [i18n("mid_side_method"), "2", 2, "mid_side_method"]:
            apollo_method = "mid_side_method"
        elif apollo_method in [i18n("normal_method"), "1", 1, "normal_method"]:
            apollo_method = "normal_method"
        else:
            print(f"Invalid apollo_method: {apollo_method}. Defaulting to normal_method.")
            apollo_method = "normal_method"
        print(f"Interpreted apollo_method: {apollo_method}")

        print(f"Corrected values - inference_chunk_size: {inference_chunk_size}, inference_overlap: {inference_overlap}, apollo_chunk_size: {apollo_chunk_size}, apollo_overlap: {apollo_overlap}")

        # Get model config using cleaned model name
        model_type, config_path, start_check_point = get_model_config(clean_model_name, inference_chunk_size, inference_overlap)
        print(f"Model configuration: model_type={model_type}, config_path={config_path}, start_check_point={start_check_point}")

        outputs = run_command_and_process_files(
            model_type=model_type,
            config_path=config_path,
            start_check_point=start_check_point,
            INPUT_DIR=INPUT_DIR,
            OUTPUT_DIR=OUTPUT_DIR,
            extract_instrumental=extract_instrumental,
            use_tta=use_tta,
            demud_phaseremix_inst=demud_phaseremix_inst,
            progress=progress,
            use_apollo=use_apollo,
            apollo_normal_model=apollo_normal_model,
            inference_chunk_size=inference_chunk_size,
            inference_overlap=inference_overlap,
            apollo_chunk_size=apollo_chunk_size,
            apollo_overlap=apollo_overlap,
            apollo_method=apollo_method,
            apollo_midside_model=apollo_midside_model,
            output_format=export_format.split()[0].lower()
        )

        if outputs is None or all(output is None for output in outputs):
            raise ValueError("run_command_and_process_files returned None or all None outputs")

        return (
            outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6],
            outputs[7], outputs[8], outputs[9], outputs[10], outputs[11], outputs[12], outputs[13],
            i18n("audio_processing_completed"),
            update_progress_html(i18n("audio_processing_completed_progress_label"), 100)
        )

    except Exception as e:
        print(f"Error in process_audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return (
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            i18n("error_occurred").format(str(e)),
            update_progress_html(i18n("error_occurred_progress_label"), 0)
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
        total_estimated_time = 10.0  # Adjust based on actual ensemble time
        elapsed_time = 0
        while elapsed_time < total_estimated_time:
            elapsed_time = time.time() - start_time
            progress_value = (elapsed_time / total_estimated_time) * 100
            progress_value = clamp_percentage(progress_value)
            progress(progress_value, desc=i18n("ensembling_progress").format(progress_value))
            time.sleep(0.1)
        
        progress(100, desc=i18n("finalizing_ensemble_output"))
        log = i18n("success_log").format(result.stdout) if not result.stderr else i18n("error_log").format(result.stderr)
        return output_path, log

    except Exception as e:
        return None, i18n("critical_error").format(str(e))
    finally:
        progress(100, desc=i18n("ensemble_process_completed"))

def auto_ensemble_process(
    auto_input_audio_file,
    selected_models,
    auto_chunk_size,
    auto_overlap,
    export_format,
    auto_use_tta,
    auto_extract_instrumental,
    auto_ensemble_type,
    _state,
    auto_use_apollo=True,
    auto_apollo_normal_model="Apollo Universal Model",
    auto_apollo_chunk_size=19,
    auto_apollo_overlap=2,
    auto_apollo_method="normal_method",
    progress=gr.Progress(track_tqdm=True),
    *args,
    **kwargs
):
    try:
        if not selected_models or len(selected_models) < 1:
            yield None, i18n("no_models_selected"), update_progress_html(i18n("error_occurred_progress_label"), 0)

        if auto_input_audio_file is None:
            existing_files = os.listdir(INPUT_DIR)
            if not existing_files:
                yield None, i18n("no_input_audio_provided"), update_progress_html(i18n("error_occurred_progress_label"), 0)
            audio_path = os.path.join(INPUT_DIR, existing_files[0])
        else:
            audio_path = auto_input_audio_file.name if hasattr(auto_input_audio_file, 'name') else auto_input_audio_file

        print(f"Raw UI inputs - auto_apollo_chunk_size: {auto_apollo_chunk_size}, auto_apollo_overlap: {auto_apollo_overlap}")
        if auto_apollo_method == "2" or auto_apollo_method == 2:
            auto_apollo_method = "mid_side_method"
        elif auto_apollo_method == "1" or auto_apollo_method == 1:
            auto_apollo_method = "normal_method"
        print(f"Interpreted auto_apollo_method: {auto_apollo_method}")

        corrected_auto_chunk_size = int(auto_apollo_chunk_size)
        corrected_auto_overlap = int(auto_apollo_overlap)
        print(f"Corrected values - auto_apollo_chunk_size: {corrected_auto_chunk_size}, auto_apollo_overlap: {corrected_auto_overlap}")

        auto_ensemble_temp = os.path.join(BASE_DIR, "auto_ensemble_temp")
        os.makedirs(auto_ensemble_temp, exist_ok=True)
        os.makedirs(AUTO_ENSEMBLE_OUTPUT, exist_ok=True)
        clear_directory(auto_ensemble_temp)
        clear_directory(AUTO_ENSEMBLE_OUTPUT)

        all_outputs = []
        total_models = len(selected_models)
        model_progress_range = 60
        model_progress_per_step = model_progress_range / total_models if total_models > 0 else 0

        for i, model in enumerate(selected_models):
            clean_model_name = clean_model(model)  # Use clean_model to remove ⭐
            print(f"Processing model {i+1}/{total_models}: Original={model}, Cleaned={clean_model_name}")
            model_output_dir = os.path.join(auto_ensemble_temp, clean_model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            current_progress = i * model_progress_per_step
            current_progress = clamp_percentage(current_progress)
            yield None, i18n("loading_model").format(i+1, total_models, clean_model_name), update_progress_html(
                i18n("loading_model_progress_label").format(i+1, total_models, clean_model_name, current_progress),
                current_progress
            )

            model_type, config_path, start_check_point = get_model_config(clean_model_name, auto_chunk_size, auto_overlap)
            print(f"Model config: model_type={model_type}, config_path={config_path}, start_check_point={start_check_point}")

            cmd = [
                "python", INFERENCE_PATH,
                "--model_type", model_type,
                "--config_path", config_path,
                "--start_check_point", start_check_point,
                "--input_folder", INPUT_DIR,
                "--store_dir", model_output_dir,
            ]
            if auto_use_tta:
                cmd.append("--use_tta")
            if auto_extract_instrumental:
                cmd.append("--extract_instrumental")

            print(i18n("running_command").format(' '.join(cmd)))
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            stderr_output = ""
            for line in process.stdout:
                print(line.strip())
                if "Progress:" in line:
                    try:
                        percentage = float(re.search(r"Progress: (\d+\.\d+)%", line).group(1))
                        model_percentage = (percentage / 100) * model_progress_per_step
                        current_progress = (i * model_progress_per_step) + model_percentage
                        current_progress = clamp_percentage(current_progress)
                        yield None, i18n("loading_model").format(i+1, total_models, clean_model_name), update_progress_html(
                            i18n("loading_model_progress_label").format(i+1, total_models, clean_model_name, current_progress),
                            current_progress
                        )
                    except (AttributeError, ValueError) as e:
                        print(i18n("progress_parsing_error").format(e))

            for line in process.stderr:
                stderr_output += line
                print(line.strip())

            process.wait()
            if process.returncode != 0:
                print(i18n("error").format(stderr_output))
                yield None, i18n("model_failed").format(clean_model_name, stderr_output), update_progress_html(
                    i18n("error_occurred_progress_label"), 0
                )
                return

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            current_progress = (i + 1) * model_progress_per_step
            current_progress = clamp_percentage(current_progress)
            yield None, i18n("completed_model").format(i+1, total_models, clean_model_name), update_progress_html(
                i18n("completed_model_progress_label").format(i+1, total_models, clean_model_name, current_progress),
                current_progress
            )

            model_outputs = glob.glob(os.path.join(model_output_dir, "*.wav"))
            if not model_outputs:
                raise FileNotFoundError(i18n("model_output_failed").format(clean_model_name))
            all_outputs.extend(model_outputs)

        # Select compatible files for ensemble (e.g., all 'instrumental' or all 'vocals')
        preferred_type = 'instrumental' if auto_extract_instrumental else 'vocals'
        ensemble_files = []
        for output in all_outputs:
            if preferred_type.lower() in output.lower():
                ensemble_files.append(output)
        print(f"Selected ensemble files: {ensemble_files}")
        if len(ensemble_files) < 2:
            print(f"Warning: Insufficient {preferred_type} files ({len(ensemble_files)}). Falling back to all outputs.")
            ensemble_files = all_outputs
            if len(ensemble_files) < 2:
                raise ValueError(i18n("insufficient_files_for_ensemble").format(len(ensemble_files)))

        # Enhanced outputs with Apollo (if enabled)
        enhanced_outputs = []
        if auto_use_apollo:
            apollo_script = "/content/Apollo/inference.py"
            print(f"Apollo parameters - chunk_size: {corrected_auto_chunk_size}, overlap: {corrected_auto_overlap}, method: {auto_apollo_method}, normal_model: {auto_apollo_normal_model}")

            # Allocate 30% for Apollo enhancement (60-90%)
            apollo_progress_range = 30  # 60-90%
            total_outputs = len(all_outputs)
            apollo_progress_per_file = apollo_progress_range / total_outputs if total_outputs > 0 else 0

            yield None, i18n("auto_enhancing_with_apollo").format(0, total_outputs), update_progress_html(
                i18n("waiting_for_files_progress_label"), 60
            )

            if auto_apollo_method == "mid_side_method":
                ckpt = "/content/Apollo/model/apollo_universal_model.ckpt"
                config = "/content/Apollo/configs/config_apollo.yaml"
            else:
                if auto_apollo_normal_model == "MP3 Enhancer":
                    ckpt = "/content/Apollo/model/pytorch_model.bin"
                    config = "/content/Apollo/configs/apollo.yaml"
                elif auto_apollo_normal_model == "Lew Vocal Enhancer":
                    ckpt = "/content/Apollo/model/apollo_model.ckpt"
                    config = "/content/Apollo/configs/apollo.yaml"
                elif auto_apollo_normal_model == "Lew Vocal Enhancer v2 (beta)":
                    ckpt = "/content/Apollo/model/apollo_model_v2.ckpt"
                    config = "/content/Apollo/configs/config_apollo_vocal.yaml"
                else:
                    ckpt = "/content/Apollo/model/apollo_universal_model.ckpt"
                    config = "/content/Apollo/configs/config_apollo.yaml"

            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"Apollo checkpoint file not found: {ckpt}")
            if not os.path.exists(config):
                raise FileNotFoundError(f"Apollo config file not found: {config}")

            for idx, output_file in enumerate(all_outputs):
                original_file_name = sanitize_filename(os.path.splitext(os.path.basename(output_file))[0])
                enhancement_suffix = "_Mid_Side_Enhanced" if auto_apollo_method == "mid_side_method" else "_Enhanced"
                enhanced_output = os.path.join(auto_ensemble_temp, f"{original_file_name}{enhancement_suffix}.wav")

                current_progress = 60 + (idx * apollo_progress_per_file)
                current_progress = clamp_percentage(current_progress)
                yield None, i18n("auto_enhancing_with_apollo").format(idx+1, total_outputs), update_progress_html(
                    i18n("enhancing_with_apollo").format(idx+1, total_outputs, original_file_name),
                    current_progress
                )

                if auto_apollo_method == "mid_side_method":
                    audio, sr = librosa.load(output_file, mono=False, sr=None)
                    if audio.ndim == 1:
                        audio = np.array([audio, audio])

                    mid = (audio[0] + audio[1]) * 0.5
                    side = (audio[0] - audio[1]) * 0.5

                    mid_file = os.path.join(auto_ensemble_temp, f"{original_file_name}_mid_temp.wav")
                    side_file = os.path.join(auto_ensemble_temp, f"{original_file_name}_side_temp.wav")
                    sf.write(mid_file, mid, sr)
                    sf.write(side_file, side, sr)

                    mid_output = os.path.join(auto_ensemble_temp, f"{original_file_name}_mid_enhanced.wav")
                    command_mid = [
                        "python", apollo_script,
                        "--in_wav", mid_file,
                        "--out_wav", mid_output,
                        "--ckpt", ckpt,
                        "--config", config,
                        "--chunk_size", str(corrected_auto_chunk_size),
                        "--overlap", str(corrected_auto_overlap)
                    ]
                    print(f"Running Mid Apollo command with chunk_size={corrected_auto_chunk_size}, overlap={corrected_auto_overlap}: {' '.join(command_mid)}")
                    result_mid = subprocess.run(command_mid, capture_output=True, text=True)
                    if result_mid.returncode != 0:
                        print(f"Apollo Mid processing failed: {result_mid.stderr}")
                        enhanced_outputs.append(output_file)
                        continue

                    side_output = os.path.join(auto_ensemble_temp, f"{original_file_name}_side_enhanced.wav")
                    command_side = [
                        "python", apollo_script,
                        "--in_wav", side_file,
                        "--out_wav", side_output,
                        "--ckpt", ckpt,
                        "--config", config,
                        "--chunk_size", str(corrected_auto_chunk_size),
                        "--overlap", str(corrected_auto_overlap)
                    ]
                    print(f"Running Side Apollo command with chunk_size={corrected_auto_chunk_size}, overlap={corrected_auto_overlap}: {' '.join(command_side)}")
                    result_side = subprocess.run(command_side, capture_output=True, text=True)
                    if result_side.returncode != 0:
                        print(f"Apollo Side processing failed: {result_side.stderr}")
                        enhanced_outputs.append(output_file)
                        continue

                    mid_audio, _ = librosa.load(mid_output, sr=sr, mono=True)
                    side_audio, _ = librosa.load(side_output, sr=sr, mono=True)
                    left = mid_audio + side_audio
                    right = mid_audio - side_audio
                    combined = np.array([left, right])
                    sf.write(enhanced_output, combined.T, sr)

                    for temp_file in [mid_file, side_file, mid_output, side_output]:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)

                    enhanced_outputs.append(enhanced_output)
                else:
                    command = [
                        "python", apollo_script,
                        "--in_wav", output_file,
                        "--out_wav", enhanced_output,
                        "--ckpt", ckpt,
                        "--config", config,
                        "--chunk_size", str(corrected_auto_chunk_size),
                        "--overlap", str(corrected_auto_overlap)
                    ]
                    print(f"Running Normal Apollo command with chunk_size={corrected_auto_chunk_size}, overlap={corrected_auto_overlap}: {' '.join(command)}")
                    apollo_process = subprocess.Popen(
                        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                    )
                    stdout_output = ""
                    for line in apollo_process.stdout:
                        print(f"Apollo Enhancing {original_file_name}: {line.strip()}")
                        stdout_output += line
                    apollo_process.wait()

                    if apollo_process.returncode != 0:
                        print(f"Apollo failed for {output_file}: {stdout_output}")
                        enhanced_outputs.append(output_file)
                        continue

                    enhanced_outputs.append(enhanced_output)

                current_progress = 60 + ((idx + 1) * apollo_progress_per_file)
                current_progress = clamp_percentage(current_progress)
                yield None, i18n("auto_enhancing_with_apollo").format(idx+1, total_outputs), update_progress_html(
                    i18n("enhancing_with_apollo").format(idx+1, total_outputs, original_file_name),
                    current_progress
                )

            all_outputs = enhanced_outputs

        # Allocate 10% for ensemble (90-100%)
        yield None, i18n("performing_ensemble"), update_progress_html(
            i18n("performing_ensemble_progress_label"), 90
        )

        quoted_files = [f'"{f}"' for f in ensemble_files]
        timestamp = str(int(time.time()))
        output_path = os.path.join(AUTO_ENSEMBLE_OUTPUT, f"ensemble_{timestamp}.wav")

        ensemble_cmd = [
            "python", ENSEMBLE_PATH,
            "--files", *quoted_files,
            "--type", auto_ensemble_type,
            "--output", f'"{output_path}"'
        ]

        print(i18n("memory_usage_before_ensemble").format(psutil.virtual_memory().percent))
        print(f"Running Ensemble command: {' '.join(ensemble_cmd)}")
        process = subprocess.Popen(
            " ".join(ensemble_cmd),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        start_time = time.time()
        total_estimated_time = 10.0  # Adjust based on actual ensemble time
        elapsed_time = 0
        while elapsed_time < total_estimated_time:
            elapsed_time = time.time() - start_time
            # Scale progress within 90-98%
            progress_value = 90 + ((elapsed_time / total_estimated_time) * 8)
            progress_value = clamp_percentage(progress_value)
            yield None, i18n("performing_ensemble"), update_progress_html(
                i18n("ensembling_progress").format(progress_value),
                progress_value
            )
            time.sleep(0.1)

        stdout_output = ""
        stderr_output = ""
        for line in process.stdout:
            stdout_output += line
            print(line.strip())
        for line in process.stderr:
            stderr_output += line
            print(line.strip())

        process.wait()
        if process.returncode != 0:
            print(i18n("error").format(stderr_output))
            yield None, i18n("error").format(stderr_output), update_progress_html(
                i18n("error_occurred_progress_label"), 0
            )
            return

        print(i18n("memory_usage_after_ensemble").format(psutil.virtual_memory().percent))

        yield None, i18n("finalizing_ensemble_output"), update_progress_html(
            i18n("finalizing_ensemble_output_progress_label"), 98
        )

        if not os.path.exists(output_path):
            raise RuntimeError(i18n("ensemble_file_creation_failed").format(output_path))

        yield output_path, i18n("success_output_created"), update_progress_html(
            i18n("ensemble_completed_progress_label"), 100
        )

    except Exception as e:
        print(f"Error in auto_ensemble_process: {str(e)}")
        import traceback
        traceback.print_exc()
        yield None, i18n("error").format(str(e)), update_progress_html(
            i18n("error_occurred_progress_label"), 0
        )
    finally:
        shutil.rmtree(auto_ensemble_temp, ignore_errors=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
