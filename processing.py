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
from helpers import INPUT_DIR, OLD_OUTPUT_DIR, ENSEMBLE_DIR, AUTO_ENSEMBLE_TEMP, move_old_files, clear_directory, BASE_DIR, clean_model, extract_model_name_from_checkpoint, sanitize_filename, find_clear_segment, save_segment, run_matchering, clamp_percentage
from model import get_model_config
from apollo_processing import process_with_apollo  # Import Apollo processing
import torch

# PyTorch optimized backend (always available)
try:
    from pytorch_backend import PyTorchBackend
    PYTORCH_OPTIMIZED_AVAILABLE = True
except ImportError:
    PYTORCH_OPTIMIZED_AVAILABLE = False
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
try:
    from google.colab import drive
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
import matchering as mg

import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INFERENCE_PATH = os.path.join(BASE_DIR, "inference.py")
ENSEMBLE_PATH = os.path.join(BASE_DIR, "ensemble.py")

if IS_COLAB:
    AUTO_ENSEMBLE_OUTPUT = "/content/drive/MyDrive/ensemble_output"
    OUTPUT_DIR = "/content/drive/MyDrive/!output_file"
else:
    AUTO_ENSEMBLE_OUTPUT = os.path.join(BASE_DIR, "ensemble_output")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(AUTO_ENSEMBLE_OUTPUT, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def setup_directories():
    """Create necessary directories and check Google Drive access."""
    if IS_COLAB:
        try:
            if not os.path.exists('/content/drive/MyDrive'):
                print("Mounting Google Drive...")
                from google.colab import drive
                drive.mount('/content/drive', force_remount=True)
            if not os.path.exists('/content/drive/MyDrive'):
                raise RuntimeError("Google Drive mount failed. Please mount manually with 'from google.colab import drive; drive.mount('/content/drive', force_remount=True)'.")
        except Exception as e:
            raise RuntimeError(f"Failed to mount Google Drive: {str(e)}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OLD_OUTPUT_DIR, exist_ok=True)
    os.makedirs(AUTO_ENSEMBLE_OUTPUT, exist_ok=True)

def refresh_auto_output():
    try:
        output_files = glob.glob(os.path.join(AUTO_ENSEMBLE_OUTPUT, "*.wav"))
        if not output_files:
            return None, "No output files found"
        
        latest_file = max(output_files, key=os.path.getctime)
        return latest_file, "Output refreshed successfully"
    except Exception as e:
        return None, f"Error refreshing output: {str(e)}"

def update_progress_html(progress_label, progress_percent):
    """Generate progress HTML."""
    progress_percent = clamp_percentage(progress_percent)
    if progress_percent > 100:
        print(f"Warning: Progress percentage {progress_percent} exceeds 100, clamping to 100")
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
    output_format="wav",
    use_pytorch_optimized=False,
    optimize_mode='default'
):
    """
    Run inference.py with specified parameters and process output files.
    """
    try:
        # Create directories and check Google Drive access
        setup_directories()

        print(f"run_command_and_process_files: model_type={model_type}, config_path={config_path}, start_check_point={start_check_point}, inference_chunk_size={inference_chunk_size}, inference_overlap={inference_overlap}, apollo_chunk_size={apollo_chunk_size}, apollo_overlap={apollo_overlap}, progress_type={type(progress)}")
        if not config_path:
            raise ValueError(f"Configuration path is empty: model_type: {model_type}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        if not start_check_point or not os.path.exists(start_check_point):
            raise FileNotFoundError(f"Checkpoint file not found: {start_check_point}")

        # Validate inference parameters
        try:
            inference_chunk_size = int(inference_chunk_size)
            inference_overlap = int(inference_overlap)
        except (TypeError, ValueError) as e:
            print(f"Invalid inference_chunk_size or inference_overlap: {e}. Defaulting to: inference_chunk_size=352800, inference_overlap=2")
            inference_chunk_size = 352800
            inference_overlap = 2

        # Validate Apollo parameters
        try:
            apollo_chunk_size = int(apollo_chunk_size)
            apollo_overlap = int(apollo_overlap)
        except (TypeError, ValueError) as e:
            print(f"Invalid apollo_chunk_size or apollo_overlap: {e}. Defaulting to: apollo_chunk_size=19, apollo_overlap=2")
            apollo_chunk_size = 19
            apollo_overlap = 2

                # Check which backend to use
        python_exe = "python"
        
        if use_pytorch_optimized and PYTORCH_OPTIMIZED_AVAILABLE:
            from inference_pytorch import INFERENCE_PATH as PYTORCH_INFERENCE_PATH
            inference_script = PYTORCH_INFERENCE_PATH if os.path.exists(PYTORCH_INFERENCE_PATH) else INFERENCE_PATH
            print(f"🔥 Using optimized PyTorch backend (mode: {optimize_mode})")
        else:
            inference_script = INFERENCE_PATH
        
        cmd_parts = [
            python_exe, inference_script,
            "--model_type", model_type,
            "--config_path", config_path,
            "--start_check_point", start_check_point,
            "--input_folder", INPUT_DIR,
            "--store_dir", OUTPUT_DIR,
            "--chunk_size", str(inference_chunk_size),
            "--overlap", str(inference_overlap),
            "--export_format", f"{output_format} FLOAT"
        ]
        
        # Add backend-specific arguments
        if use_pytorch_optimized and PYTORCH_OPTIMIZED_AVAILABLE:
            cmd_parts.extend([
                "--optimize_mode", optimize_mode,
                "--enable_amp",
                "--enable_tf32",
                "--enable_cudnn_benchmark"
            ])
        
        if extract_instrumental:
            cmd_parts.append("--extract_instrumental")
        if use_tta:
            cmd_parts.append("--use_tta")
        if demud_phaseremix_inst:
            cmd_parts.append("--demud_phaseremix_inst")

        print(f"Running command: {' '.join(cmd_parts)}")
        
        try:
            process = subprocess.run(
                cmd_parts,
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                check=True,
                timeout=3600  # 1 hour timeout
            )
        except subprocess.CalledProcessError as e:
            # If TensorRT fails, fallback to PyTorch
            if use_tensorrt and "CUDA initialization failure" in str(e.stderr):
                print("⚠️  TensorRT CUDA error, falling back to standard PyTorch")
                cmd_parts = [
                    "python", INFERENCE_PATH,
                    "--model_type", model_type,
                    "--config_path", config_path,
                    "--start_check_point", start_check_point,
                    "--input_folder", INPUT_DIR,
                    "--store_dir", OUTPUT_DIR,
                    "--chunk_size", str(inference_chunk_size),
                    "--overlap", str(inference_overlap),
                    "--export_format", f"{output_format} FLOAT"
                ]
                if extract_instrumental:
                    cmd_parts.append("--extract_instrumental")
                if use_tta:
                    cmd_parts.append("--use_tta")
                if demud_phaseremix_inst:
                    cmd_parts.append("--demud_phaseremix_inst")
                
                print(f"Retry with PyTorch: {' '.join(cmd_parts)}")
                process = subprocess.run(
                    cmd_parts,
                    cwd=BASE_DIR,
                    capture_output=True,
                    text=True,
                    check=True
                )
            else:
                raise

        # Log subprocess output
        print(f"Subprocess stdout: {process.stdout}")
        if process.stderr:
            print(f"Subprocess stderr: {process.stderr}")

        # Progress update (separation phase, 0-80%)
        if progress is not None and callable(getattr(progress, '__call__', None)):
            progress(0, desc="Starting audio separation", total=100)
        else:
            print("Progress is not callable or None, skipping progress update")

        # Check if output files were created
        filename_model = extract_model_name_from_checkpoint(start_check_point)
        output_files = os.listdir(OUTPUT_DIR)
        if not output_files:
            raise FileNotFoundError("No output files created in OUTPUT_DIR")

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
                    print(f"Could not rename {file_path} to {new_file_path}: {str(e)}")

        rename_files_with_model(OUTPUT_DIR, filename_model)

        output_files = os.listdir(OUTPUT_DIR)
        if not output_files:
            raise FileNotFoundError("No output files in OUTPUT_DIR after renaming")

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

        # Apollo processing
        if use_apollo:
            normalized_outputs = process_with_apollo(
                output_files=normalized_outputs,
                output_dir=OUTPUT_DIR,
                apollo_chunk_size=apollo_chunk_size,
                apollo_overlap=apollo_overlap,
                apollo_method=apollo_method,
                apollo_normal_model=apollo_normal_model,
                apollo_midside_model=apollo_midside_model,
                output_format=output_format,
                progress=progress,
                total_progress_start=80,
                total_progress_end=100
            )

        # Final progress update
        if progress is not None and callable(getattr(progress, '__call__', None)):
            progress(100, desc="Separation complete")
        return tuple(normalized_outputs)

    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed, code: {e.returncode}: {e.stderr}")
        return (None,) * 14
    except Exception as e:
        print(f"run_command_and_process_files error: {str(e)}")
        import traceback
        traceback.print_exc()
        return (None,) * 14

def process_audio(
    input_audio_file,
    model,
    chunk_size,
    overlap,
    export_format,
    use_pytorch_optimized,
    optimize_mode,
    use_tta,
    demud_phaseremix_inst,
    extract_instrumental,
    use_apollo,
    apollo_chunk_size,
    apollo_overlap,
    apollo_method,
    apollo_normal_model,
    apollo_midside_model,
    use_matchering,
    matchering_passes,
    progress=gr.Progress(track_tqdm=True),
    *args,
    **kwargs
):
    try:
        # Check Google Drive connection
        setup_directories()

        if input_audio_file is not None:
            audio_path = input_audio_file.name if hasattr(input_audio_file, 'name') else input_audio_file
        else:
            return (
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                "No audio file provided",
                update_progress_html("No input provided", 0)
            )

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(OLD_OUTPUT_DIR, exist_ok=True)
        move_old_files(OUTPUT_DIR)

        print(f"process_audio: model parameter received: {model}")
        # Clean model name, remove ⭐ and other unwanted characters
        clean_model_name = clean_model(model) if not model.startswith("/") else extract_model_name_from_checkpoint(model)
        print(f"Processing audio: {audio_path}, model: {clean_model_name}")

        print(f"Raw UI inputs - chunk_size: {chunk_size}, overlap: {overlap}, apollo_chunk_size: {apollo_chunk_size}, apollo_overlap: {apollo_overlap}, apollo_method: {apollo_method}")

        # Validate inference parameters
        try:
            inference_chunk_size = int(chunk_size)
        except (TypeError, ValueError):
            print(f"Invalid chunk_size: {chunk_size}. Defaulting to: 352800.")
            inference_chunk_size = 352800

        try:
            inference_overlap = int(overlap)
        except (TypeError, ValueError):
            print(f"Invalid overlap: {overlap}. Defaulting to: 2.")
            inference_overlap = 2

        # Validate Apollo parameters
        try:
            apollo_chunk_size = int(apollo_chunk_size)
        except (TypeError, ValueError):
            print(f"Invalid apollo_chunk_size: {apollo_chunk_size}. Defaulting to: 19.")
            apollo_chunk_size = 19

        try:
            apollo_overlap = int(apollo_overlap)
        except (TypeError, ValueError):
            print(f"Invalid apollo_overlap: {apollo_overlap}. Defaulting to: 2.")
            apollo_overlap = 2

        # Map apollo_method to backend values
        if apollo_method in ["Mid-side method", "2", 2, "mid_side_method"]:
            apollo_method = "mid_side_method"
        elif apollo_method in ["Normal method", "1", 1, "normal_method"]:
            apollo_method = "normal_method"
        else:
            print(f"Invalid apollo_method: {apollo_method}. Defaulting to: normal_method.")
            apollo_method = "normal_method"
        print(f"Parsed apollo_method: {apollo_method}")

        print(f"Corrected values - inference_chunk_size: {inference_chunk_size}, inference_overlap: {inference_overlap}, apollo_chunk_size: {apollo_chunk_size}, apollo_overlap: {apollo_overlap}")

        # Copy input file to INPUT_DIR
        input_filename = os.path.basename(audio_path)
        dest_path = os.path.join(INPUT_DIR, input_filename)
        shutil.copy(audio_path, dest_path)
        print(f"Input file copied: {dest_path}")

        # Get model configuration with cleaned model name
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
            output_format=export_format.split()[0].lower(),
            use_pytorch_optimized=use_pytorch_optimized,
            optimize_mode=optimize_mode
        )

        if outputs is None or all(output is None for output in outputs):
            raise ValueError("run_command_and_process_files returned None or all None outputs")

        # Apply Matchering (if enabled)
        if use_matchering:
            # Progress update for Matchering
            if progress is not None and callable(getattr(progress, '__call__', None)):
                progress(90, desc="Applying Matchering")

            # Find clean segment from original audio
            segment_start, segment_end, segment_audio = find_clear_segment(audio_path)
            segment_path = os.path.join(tempfile.gettempdir(), "matchering_segment.wav")
            save_segment(segment_audio, 44100, segment_path)

            # Process each output with Matchering
            mastered_outputs = []
            for output in outputs:
                if output and os.path.exists(output):
                    output_base = sanitize_filename(os.path.splitext(os.path.basename(output))[0])
                    mastered_path = os.path.join(OUTPUT_DIR, f"{output_base}_mastered.wav")
                    mastered_output = run_matchering(
                        reference_path=segment_path,
                        target_path=output,
                        output_path=mastered_path,
                        passes=matchering_passes,
                        bit_depth=24
                    )
                    mastered_outputs.append(mastered_path)
                else:
                    mastered_outputs.append(output)

            # Clean up segment file
            if os.path.exists(segment_path):
                os.remove(segment_path)

            outputs = tuple(mastered_outputs)

        if progress is not None and callable(getattr(progress, '__call__', None)):
            progress(100, desc="Processing complete")

        return (
            outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6],
            outputs[7], outputs[8], outputs[9], outputs[10], outputs[11], outputs[12], outputs[13],
            "Audio processing completed",
            update_progress_html("Audio processing completed", 100)
        )

    except Exception as e:
        print(f"process_audio error: {str(e)}")
        import traceback
        traceback.print_exc()
        return (
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            f"Error occurred: {str(e)}",
            update_progress_html("Error occurred", 0)
        )

def ensemble_audio_fn(files, method, weights, progress=gr.Progress()):
    try:
        if len(files) < 2:
            return None, "Minimum two files required"
        
        valid_files = [f for f in files if os.path.exists(f)]
        if len(valid_files) < 2:
            return None, "Valid files not found"
        
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
        
        progress(0, desc="Starting ensemble process", total=100)
        result = subprocess.run(
            ["python", "ensemble.py"] + ensemble_args,
            capture_output=True,
            text=True
        )
        
        start_time = time.time()
        total_estimated_time = 10.0  # Adjust based on actual ensemble duration
        elapsed_time = 0
        while elapsed_time < total_estimated_time:
            elapsed_time = time.time() - start_time
            progress_value = (elapsed_time / total_estimated_time) * 100
            progress_value = clamp_percentage(progress_value)
            progress(progress_value, desc=f"Ensembling progress: {progress_value}%")
            time.sleep(0.1)
        
        progress(100, desc="Finalizing ensemble output")
        log = f"Success: {result.stdout}" if not result.stderr else f"Error: {result.stderr}"
        return output_path, log

    except Exception as e:
        return None, f"Critical error: {str(e)}"
    finally:
        progress(100, desc="Ensemble process completed")


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
    auto_use_matchering=False,
    auto_matchering_passes=1,
    apollo_midside_model=None,
    progress=gr.Progress(track_tqdm=True)
):
    """Process audio with multiple models and ensemble the results, saving output to Google Drive."""
    try:
        # Check Google Drive connection and setup directories
        setup_directories()

        if not selected_models or len(selected_models) < 1:
            yield None, i18n("no_models_selected"), update_progress_html(i18n("error_occurred"), 0)
            return

        if auto_input_audio_file is None:
            existing_files = os.listdir(INPUT_DIR)
            if not existing_files:
                yield None, i18n("no_input_audio_provided"), update_progress_html(i18n("error_occurred"), 0)
                return
            audio_path = os.path.join(INPUT_DIR, existing_files[0])
        else:
            audio_path = auto_input_audio_file.name if hasattr(auto_input_audio_file, 'name') else auto_input_audio_file

        # Copy input file to INPUT_DIR
        input_filename = os.path.basename(audio_path)
        dest_path = os.path.join(INPUT_DIR, input_filename)
        shutil.copy(audio_path, dest_path)
        print(f"Input file copied: {dest_path}")

        # Parse apollo method
        if auto_apollo_method in ["2", 2]:
            auto_apollo_method = "mid_side_method"
        elif auto_apollo_method in ["1", 1]:
            auto_apollo_method = "normal_method"
        print(f"Parsed auto_apollo_method: {auto_apollo_method}")

        corrected_auto_chunk_size = int(auto_apollo_chunk_size)
        corrected_auto_overlap = int(auto_apollo_overlap)
        print(f"Corrected values - auto_apollo_chunk_size: {corrected_auto_chunk_size}, auto_apollo_overlap: {corrected_auto_overlap}")

        # Setup temporary directories
        auto_ensemble_temp = os.path.join(BASE_DIR, "auto_ensemble_temp")
        os.makedirs(auto_ensemble_temp, exist_ok=True)
        clear_directory(auto_ensemble_temp)

        all_outputs = []
        total_models = len(selected_models)
        model_progress_range = 60
        model_progress_per_step = model_progress_range / total_models if total_models > 0 else 0

        for i, model in enumerate(selected_models):
            clean_model_name = clean_model(model)
            print(f"Processing model {i+1}/{total_models}: Original={model}, Cleaned={clean_model_name}")
            model_output_dir = os.path.join(auto_ensemble_temp, clean_model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            current_progress = i * model_progress_per_step
            current_progress = clamp_percentage(current_progress)
            yield None, i18n("loading_model").format(i+1, total_models, clean_model_name), update_progress_html(
                i18n("loading_model_progress").format(i+1, total_models, clean_model_name, current_progress),
                current_progress
            )

            model_type, config_path, start_check_point = get_model_config(clean_model_name, auto_chunk_size, auto_overlap)
            print(f"Model configuration: model_type={model_type}, config_path={config_path}, start_check_point={start_check_point}")

            cmd = [
                "python", INFERENCE_PATH,
                "--model_type", model_type,
                "--config_path", config_path,
                "--start_check_point", start_check_point,
                "--input_folder", INPUT_DIR,
                "--store_dir", model_output_dir,
                "--chunk_size", str(auto_chunk_size),
                "--overlap", str(auto_overlap),
                "--export_format", f"{export_format.split()[0].lower()} FLOAT"
            ]
            if auto_use_tta:
                cmd.append("--use_tta")
            if auto_extract_instrumental:
                cmd.append("--extract_instrumental")

            print(f"Running command: {' '.join(cmd)}")
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
                            i18n("loading_model_progress").format(i+1, total_models, clean_model_name, current_progress),
                            current_progress
                        )
                    except (AttributeError, ValueError) as e:
                        print(f"Progress parsing error: {e}")

            for line in process.stderr:
                stderr_output += line
                print(line.strip())

            process.wait()
            if process.returncode != 0:
                print(f"Error: {stderr_output}")
                yield None, i18n("model_failed").format(clean_model_name, stderr_output), update_progress_html(
                    i18n("error_occurred"), 0
                )
                return

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            current_progress = (i + 1) * model_progress_per_step
            current_progress = clamp_percentage(current_progress)
            yield None, i18n("completed_model").format(i+1, total_models, clean_model_name), update_progress_html(
                i18n("completed_model_progress").format(i+1, total_models, clean_model_name, current_progress),
                current_progress
            )

            model_outputs = glob.glob(os.path.join(model_output_dir, "*.wav"))
            if not model_outputs:
                raise FileNotFoundError(i18n("model_output_failed").format(clean_model_name))
            all_outputs.extend(model_outputs)

        # Select compatible files for ensemble
        preferred_type = 'instrumental' if auto_extract_instrumental else 'vocals'
        ensemble_files = [output for output in all_outputs if preferred_type.lower() in output.lower()]
        print(f"Selected ensemble files: {ensemble_files}")
        if len(ensemble_files) < 2:
            print(f"Warning: Insufficient {preferred_type} files ({len(ensemble_files)}). Falling back to all outputs.")
            ensemble_files = all_outputs
            if len(ensemble_files) < 2:
                raise ValueError(i18n("insufficient_files_for_ensemble").format(len(ensemble_files)))

        # Enhanced outputs with Apollo (if enabled)
        if auto_use_apollo:
            yield None, i18n("enhancing_with_apollo").format(0, len(all_outputs)), update_progress_html(
                i18n("waiting_for_files"), 60
            )

            all_outputs = process_with_apollo(
                output_files=all_outputs,
                output_dir=auto_ensemble_temp,
                apollo_chunk_size=corrected_auto_chunk_size,
                apollo_overlap=corrected_auto_overlap,
                apollo_method=auto_apollo_method,
                apollo_normal_model=auto_apollo_normal_model,
                apollo_midside_model=apollo_midside_model,
                output_format=export_format.split()[0].lower(),
                progress=progress,
                total_progress_start=60,
                total_progress_end=90
            )

        # Perform ensemble
        yield None, i18n("performing_ensemble"), update_progress_html(
            i18n("performing_ensemble"), 90
        )

        quoted_files = [f'"{f}"' for f in ensemble_files]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(AUTO_ENSEMBLE_OUTPUT, f"auto_ensemble_output_{timestamp}.wav")

        ensemble_cmd = [
            "python", ENSEMBLE_PATH,
            "--files", *quoted_files,
            "--type", auto_ensemble_type,
            "--output", f'"{output_path}"'
        ]

        print(f"Running ensemble command: {' '.join(ensemble_cmd)}")
        try:
            process = subprocess.Popen(
                " ".join(ensemble_cmd),
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            stdout_output = ""
            stderr_output = ""
            for line in process.stdout:
                stdout_output += line
                print(f"Ensemble stdout: {line.strip()}")
            for line in process.stderr:
                stderr_output += line
                print(f"Ensemble stderr: {line.strip()}")

            process.wait()
            if process.returncode != 0:
                print(f"Ensemble subprocess failed with code {process.returncode}: {stderr_output}")
                yield None, i18n("ensemble_error").format(stderr_output), update_progress_html(
                    i18n("error_occurred"), 0
                )
                return

            print(f"Checking if output file exists: {output_path}")
            if not os.path.exists(output_path):
                raise RuntimeError(f"Ensemble output file not created at {output_path}. Stdout: {stdout_output}, Stderr: {stderr_output}")

        except Exception as e:
            print(f"Ensemble command execution failed: {str(e)}")
            yield None, i18n("ensemble_error").format(str(e)), update_progress_html(
                i18n("error_occurred"), 0
            )
            return

        # Apply Matchering (if enabled)
        if auto_use_matchering and os.path.exists(output_path):
            yield None, i18n("applying_matchering"), update_progress_html(
                i18n("applying_matchering"), 98
            )

            try:
                # Find clean segment
                segment_start, segment_end, segment_audio = find_clear_segment(audio_path)
                segment_path = os.path.join(tempfile.gettempdir(), "matchering_segment.wav")
                save_segment(segment_audio, 44100, segment_path)

                # Master the ensemble output
                mastered_output_path = os.path.join(AUTO_ENSEMBLE_OUTPUT, f"auto_ensemble_output_{timestamp}_mastered.wav")
                print(f"Running Matchering: reference={segment_path}, target={output_path}, output={mastered_output_path}")
                mastered_output = run_matchering(
                    reference_path=segment_path,
                    target_path=output_path,
                    output_path=mastered_output_path,
                    passes=auto_matchering_passes,
                    bit_depth=24
                )

                # Verify mastered output
                if not os.path.exists(mastered_output_path):
                    raise RuntimeError(f"Matchering failed to create output at {mastered_output_path}")

                # Clean up segment file
                if os.path.exists(segment_path):
                    os.remove(segment_path)

                output_path = mastered_output_path
                print(f"Matchering completed: {mastered_output_path}")
            except Exception as e:
                print(f"Matchering error: {str(e)}")
                yield None, i18n("error").format(f"Matchering failed: {str(e)}"), update_progress_html(
                    i18n("error_occurred"), 0
                )
                return

        yield None, i18n("finalizing_ensemble_output"), update_progress_html(
            i18n("finalizing_ensemble_output"), 98
        )

        if not os.path.exists(output_path):
            raise RuntimeError(i18n("ensemble_file_creation_failed").format(output_path))

        # Verify write permissions for Google Drive directory
        try:
            print(f"Verifying write permissions for {AUTO_ENSEMBLE_OUTPUT}")
            test_file = os.path.join(AUTO_ENSEMBLE_OUTPUT, "test_write.txt")
            with open(test_file, "w") as f:
                f.write("Test")
            os.remove(test_file)
            print(f"Write permissions verified for {AUTO_ENSEMBLE_OUTPUT}")
        except Exception as e:
            print(f"Write permission error for {AUTO_ENSEMBLE_OUTPUT}: {str(e)}")
            yield None, i18n("error").format(f"Write permission error: {str(e)}"), update_progress_html(
                i18n("error_occurred"), 0
            )
            return

        # Verify file in Google Drive
        print(f"Final output file: {output_path}")
        if IS_COLAB:
            drive_output_path = os.path.join("/content/drive/MyDrive/ensemble_output", os.path.basename(output_path))
            print(f"Checking if file exists in Google Drive: {drive_output_path}")
            if not os.path.exists(drive_output_path):
                print(f"File not found in Google Drive, copying from local path: {output_path}")
                shutil.copy(output_path, drive_output_path)
                print(f"Copied to Google Drive: {drive_output_path}")

        yield output_path, i18n("success_output_created") + f" Saved to {drive_output_path if IS_COLAB else output_path}", update_progress_html(
            i18n("ensemble_completed"), 100
        )

    except Exception as e:
        print(f"auto_ensemble_process error: {str(e)}")
        import traceback
        traceback.print_exc()
        yield None, i18n("error").format(str(e)), update_progress_html(
            i18n("error_occurred"), 0
        )
    finally:
        shutil.rmtree(auto_ensemble_temp, ignore_errors=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
