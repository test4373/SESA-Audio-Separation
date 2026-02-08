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

# Google OAuth imports (optional - for Colab/Google Drive support)
try:
    from google.oauth2.credentials import Credentials
    GOOGLE_OAUTH_AVAILABLE = True
except ImportError:
    GOOGLE_OAUTH_AVAILABLE = False
    Credentials = None

import tempfile
from urllib.parse import urlparse, quote
try:
    from google.colab import drive
    # Verify we're actually in a working Colab environment
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
    drive = None
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
            # Check if Google Drive is already mounted
            if os.path.exists('/content/drive/MyDrive'):
                pass  # Already mounted, no action needed
            else:
                print("Mounting Google Drive...")
                try:
                    from google.colab import drive
                    drive.mount('/content/drive', force_remount=True)
                except AttributeError as ae:
                    # Handle 'NoneType' object has no attribute 'kernel' error
                    print(f"⚠️ Google Drive mount skipped (Colab kernel issue): {str(ae)}")
                    print("Continuing with local storage...")
                except Exception as mount_error:
                    print(f"⚠️ Google Drive mount failed: {str(mount_error)}")
                    print("Continuing with local storage...")
        except Exception as e:
            print(f"⚠️ Google Drive setup error: {str(e)}")
            print("Continuing without Google Drive...")
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

def update_progress_html(progress_label, progress_percent, download_info=None):
    """Generate progress HTML with smooth animations and optional download percentage.
    
    Args:
        progress_label: Text label to show above the progress bar
        progress_percent: Overall progress percentage (0-100)
        download_info: Optional dict with 'filename' and 'percent' for download progress
    """
    progress_percent = clamp_percentage(progress_percent)
    
    # Determine if processing is active for pulse animation
    is_active = 0 < progress_percent < 100
    pulse_style = "animation: progress-pulse 1.5s ease-in-out infinite;" if is_active else ""
    
    # Build download sub-bar if downloading
    download_html = ""
    if download_info and isinstance(download_info, dict):
        dl_filename = download_info.get('filename', '')
        dl_percent = clamp_percentage(download_info.get('percent', 0))
        download_html = f"""
        <div style="margin-top: 8px; padding: 8px; background: rgba(0,0,0,0.3); border-radius: 5px;">
            <div style="font-size: 0.85rem; color: #a0a0a0; margin-bottom: 4px;">📥 {dl_filename} - %{int(dl_percent)}</div>
            <div style="width: 100%; background-color: #333; border-radius: 4px; overflow: hidden;">
                <div style="width: {dl_percent}%; height: 14px; background: linear-gradient(90deg, #4ade80, #22d3ee); transition: width 0.3s ease-out; border-radius: 4px;"></div>
            </div>
        </div>
        """
    
    return f"""
    <style>
        @keyframes progress-pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.85; }}
        }}
    </style>
    <div id="custom-progress" style="margin-top: 10px;">
        <div style="font-size: 1rem; color: #C0C0C0; margin-bottom: 5px;" id="progress-label">{progress_label}</div>
        <div style="width: 100%; background-color: #444; border-radius: 5px; overflow: hidden;">
            <div id="progress-bar" style="width: {progress_percent}%; height: 20px; background: linear-gradient(90deg, #6e8efb, #a855f7); transition: width 0.5s ease-out; max-width: 100%; {pulse_style}"></div>
        </div>
        {download_html}
    </div>
    """

def extract_model_name_from_checkpoint(checkpoint_path):
    if not checkpoint_path:
        return "Unknown"
    base_name = os.path.basename(checkpoint_path)
    model_name = os.path.splitext(base_name)[0]
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
    optimize_mode='channels_last',
    enable_amp=True,
    enable_tf32=True,
    enable_cudnn_benchmark=True
):
    """
    Run inference.py with specified parameters and process output files.
    This is a generator function that yields progress updates for real-time UI feedback.
    """
    try:
        # Create directories and check Google Drive access
        setup_directories()

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

        # Initial progress yield
        yield {"progress": 0, "status": "Starting audio separation...", "outputs": None}

        # Always use optimized PyTorch backend
        python_exe = "python"
        
        if PYTORCH_OPTIMIZED_AVAILABLE:
            from inference_pytorch import INFERENCE_PATH as PYTORCH_INFERENCE_PATH
            inference_script = PYTORCH_INFERENCE_PATH if os.path.exists(PYTORCH_INFERENCE_PATH) else INFERENCE_PATH
            print(f"🔥 Using ULTRA-OPTIMIZED PyTorch backend (mode: {optimize_mode})")
            print(f"   ⚡ AMP: {enable_amp} | 🎯 TF32: {enable_tf32} | ⚙️ cuDNN: {enable_cudnn_benchmark}")
        else:
            inference_script = INFERENCE_PATH
            print("⚠️ PyTorch optimized backend not available, using standard inference")
        







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
        








                
        # Add optimized backend arguments (always enabled)
        if PYTORCH_OPTIMIZED_AVAILABLE:
            cmd_parts.extend([
                "--optimize_mode", optimize_mode
            ])
            if enable_amp:
                cmd_parts.append("--enable_amp")
            if enable_tf32:
                cmd_parts.append("--enable_tf32")
            if enable_cudnn_benchmark:
                cmd_parts.append("--enable_cudnn_benchmark")
        
        if extract_instrumental:
            cmd_parts.append("--extract_instrumental")
        if use_tta:
            cmd_parts.append("--use_tta")
        if demud_phaseremix_inst:
            cmd_parts.append("--demud_phaseremix_inst")

        print(f"Running command: {' '.join(cmd_parts)}")
        
        # Use subprocess.Popen for real-time progress capture
        process = subprocess.Popen(
            cmd_parts,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        stderr_output = ""
        last_yield_percent = -1
        downloading_file = None
        
        # Read stdout line-by-line for real-time progress updates
        for line in process.stdout:
            line_stripped = line.strip()
            
            # Check for download progress [SESA_DOWNLOAD]
            if line_stripped.startswith("[SESA_DOWNLOAD]"):
                try:
                    dl_info = line_stripped.replace("[SESA_DOWNLOAD]", "")
                    if dl_info.startswith("START:"):
                        downloading_file = dl_info.replace("START:", "")
                        yield {"progress": 0, "status": f"📥 Model indiriliyor: {downloading_file}", "outputs": None}
                    elif dl_info.startswith("END:"):
                        downloading_file = None
                    elif ":" in dl_info:
                        parts = dl_info.rsplit(":", 1)
                        if len(parts) == 2:
                            filename, percent_str = parts
                            download_percent = int(percent_str)
                            yield {"progress": 0, "status": f"📥 İndiriliyor: {filename} - %{download_percent}", "outputs": None}
                except (ValueError, TypeError):
                    pass
            # Check for [SESA_PROGRESS] prefix from inference script
            elif line_stripped.startswith("[SESA_PROGRESS]"):
                try:
                    percentage_str = line_stripped.replace("[SESA_PROGRESS]", "").strip()
                    percentage = float(percentage_str) if percentage_str else 0
                    percentage = min(max(percentage, 0), 100)
                    
                    # Scale progress to 0-80% range (saving 80-100% for Apollo)
                    scaled_progress = int(percentage * 0.8)
                    
                    # Yield on every percent change for smooth updates
                    if int(percentage) != last_yield_percent:
                        last_yield_percent = int(percentage)
                        yield {"progress": scaled_progress, "status": f"Separating audio... {int(percentage)}%", "outputs": None}
                except (ValueError, TypeError):
                    pass
            else:
                # Only print important non-progress lines (errors, warnings, key info)
                if line_stripped and not line_stripped.startswith(("  ", "    ")):
                    print(line_stripped)
        
        # Capture stderr (only print errors)
        for line in process.stderr:
            stderr_output += line
            line_s = line.strip()
            if line_s and ("error" in line_s.lower() or "warning" in line_s.lower() or "traceback" in line_s.lower()):
                print(f"⚠️ {line_s}")
        
        process.wait()
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd_parts, stderr=stderr_output)
        
        yield {"progress": 80, "status": "Separation complete, processing outputs...", "outputs": None}

        # Check if output files were created
        filename_model = extract_model_name_from_checkpoint(start_check_point)
        output_files = os.listdir(OUTPUT_DIR)
        if not output_files:
            raise FileNotFoundError("No output files created in OUTPUT_DIR")

        def rename_files_with_model(folder, filename_model):
            timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
            for filename in sorted(os.listdir(folder)):
                file_path = os.path.join(folder, filename)
                if not any(filename.lower().endswith(ext) for ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']):
                    continue
                base, ext = os.path.splitext(filename)
                detected_type = None
                for type_key in ['vocals', 'instrumental', 'instrument', 'phaseremix', 'drum', 'bass', 'other', 'effects', 'speech', 'music', 'dry', 'male', 'female', 'bleed', 'karaoke']:
                    if type_key in base.lower():
                        detected_type = type_key
                        break
                # Normalize 'instrument' to 'Instrumental' for consistency
                type_suffix = 'Instrumental' if detected_type == 'instrument' else (detected_type.capitalize() if detected_type else "Processed")
                clean_base = sanitize_filename(base.split('_')[0]).rsplit('.', 1)[0]
                new_filename = f"{timestamp}_{clean_base}_{type_suffix}_{filename_model}{ext}"
                new_file_path = os.path.join(folder, new_filename)
                try:
                    os.rename(file_path, new_file_path)
                except Exception as e:
                    print(f"⚠️ Dosya yeniden adlandırılamadı: {os.path.basename(file_path)} -> {os.path.basename(new_file_path)}: {str(e)}")

        rename_files_with_model(OUTPUT_DIR, filename_model)

        output_files = os.listdir(OUTPUT_DIR)
        if not output_files:
            raise FileNotFoundError("No output files in OUTPUT_DIR after renaming")

        def find_file(keywords):
            """Find file matching any of the keywords (can be single keyword or list)."""
            if isinstance(keywords, str):
                keywords = [keywords]
            matching_files = [
                os.path.join(OUTPUT_DIR, f) for f in output_files 
                if any(kw in f.lower() for kw in keywords)
            ]
            return matching_files[0] if matching_files else None

        output_list = [
            find_file('vocals'), find_file(['instrumental', 'instrument']), find_file('phaseremix'),
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
            yield {"progress": 80, "status": "Enhancing with Apollo...", "outputs": None}
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

        # Final yield with outputs
        yield {"progress": 100, "status": "Separation complete", "outputs": tuple(normalized_outputs)}

    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed, code: {e.returncode}: {e.stderr}")
        yield {"progress": 0, "status": f"Error: {e.stderr}", "outputs": (None,) * 14}
    except Exception as e:
        print(f"run_command_and_process_files error: {str(e)}")
        import traceback
        traceback.print_exc()
        yield {"progress": 0, "status": f"Error: {str(e)}", "outputs": (None,) * 14}




























def process_audio(
    input_audio_file,
    model,
    chunk_size,
    overlap,
    export_format,
    optimize_mode,
    enable_amp,
    enable_tf32,
    enable_cudnn_benchmark,
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
    """
    Process audio with the selected model. This is a generator function that yields
    progress updates for real-time UI feedback.
    """
    try:
        # Check Google Drive connection
        setup_directories()

        if input_audio_file is not None:
            audio_path = input_audio_file.name if hasattr(input_audio_file, 'name') else input_audio_file
        else:
            yield (
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                "No audio file provided",
                update_progress_html("No input provided", 0)
            )
            return

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(OLD_OUTPUT_DIR, exist_ok=True)
        move_old_files(OUTPUT_DIR)

        # Clean model name, remove ⭐ and other unwanted characters
        clean_model_name = clean_model(model) if not model.startswith("/") else extract_model_name_from_checkpoint(model)
        print(f"🎵 Processing: {os.path.basename(audio_path)} | Model: {clean_model_name}")

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
        # Copy input file to INPUT_DIR
        input_filename = os.path.basename(audio_path)
        dest_path = os.path.join(INPUT_DIR, input_filename)
        shutil.copy(audio_path, dest_path)

        # Yield status for model loading
        yield (
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            f"📥 Loading model: {clean_model_name}...",
            update_progress_html(f"Loading model: {clean_model_name}", 0)
        )
        
        # Get model configuration with cleaned model name (downloads if needed)
        model_type, config_path, start_check_point = get_model_config(clean_model_name, inference_chunk_size, inference_overlap)

        # Iterate over the generator and yield progress updates
        outputs = None
        for update in run_command_and_process_files(
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
            optimize_mode=optimize_mode,
            enable_amp=enable_amp,
            enable_tf32=enable_tf32,
            enable_cudnn_benchmark=enable_cudnn_benchmark
        ):
            if update.get("outputs") is not None:
                outputs = update["outputs"]
            # Yield progress update to Gradio
            yield (
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                update["status"],
                update_progress_html(update["status"], update["progress"])
            )

        if outputs is None or all(output is None for output in outputs):
            raise ValueError("run_command_and_process_files returned None or all None outputs")

        # Apply Matchering (if enabled)
        if use_matchering:
            # Yield progress update for Matchering
            yield (
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                "Applying Matchering...",
                update_progress_html("Applying Matchering...", 90)
            )

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

        # Final yield with all outputs
        yield (
            outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6],
            outputs[7], outputs[8], outputs[9], outputs[10], outputs[11], outputs[12], outputs[13],
            "Audio processing completed",
            update_progress_html("Audio processing completed", 100)
        )

    except Exception as e:
        print(f"process_audio error: {str(e)}")
        import traceback
        traceback.print_exc()
        yield (
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
        
        # Run ensemble subprocess with real-time output capture
        process = subprocess.Popen(
            ["python", "ensemble.py"] + ensemble_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        stdout_output = ""
        stderr_output = ""
        
        # Read output in real-time and capture actual progress
        for line in process.stdout:
            stdout_output += line
            line_stripped = line.strip()
            
            # Capture real progress percentage from ensemble.py with new format
            if line_stripped.startswith("[SESA_PROGRESS]"):
                try:
                    percent_str = line_stripped.replace("[SESA_PROGRESS]", "").strip()
                    percent = int(float(percent_str)) if percent_str else 0
                    percent = min(max(percent, 0), 100)
                    progress(percent, desc=f"Ensemble progress: {percent}%")
                except (ValueError, TypeError):
                    pass
            # Legacy format support
            elif line_stripped.startswith("Progress:"):
                try:
                    percent = int(line_stripped.split(":")[1].strip().replace("%", ""))
                    percent = min(max(percent, 0), 100)
                    progress(percent, desc=f"Ensemble progress: {percent}%")
                except (ValueError, IndexError):
                    pass
            elif "loading" in line.lower():
                print(f"Ensemble: {line_stripped}")
                progress(5, desc="Loading audio files for ensemble...")
            elif "processing ensemble" in line.lower():
                print(f"Ensemble: {line_stripped}")
                progress(10, desc="Starting ensemble processing...")
            elif "saving" in line.lower():
                print(f"Ensemble: {line_stripped}")
                progress(95, desc="Saving ensemble output...")
            elif line_stripped and not line_stripped.startswith("[SESA_PROGRESS]") and not line_stripped.startswith("Progress:"):
                # Only print non-progress messages
                print(f"Ensemble: {line_stripped}")
        
        for line in process.stderr:
            stderr_output += line
            print(f"Ensemble stderr: {line.strip()}")
        
        process.wait()
        result = type('Result', (), {'stdout': stdout_output, 'stderr': stderr_output, 'returncode': process.returncode})()
        
        progress(100, desc="Ensemble complete")
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

        # Parse apollo method
        if auto_apollo_method in ["2", 2]:
            auto_apollo_method = "mid_side_method"
        elif auto_apollo_method in ["1", 1]:
            auto_apollo_method = "normal_method"

        corrected_auto_chunk_size = int(auto_apollo_chunk_size)
        corrected_auto_overlap = int(auto_apollo_overlap)

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
            model_output_dir = os.path.join(auto_ensemble_temp, clean_model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            current_progress = i * model_progress_per_step
            current_progress = clamp_percentage(current_progress)
            yield None, i18n("loading_model").format(i+1, total_models, clean_model_name), update_progress_html(
                i18n("loading_model_progress").format(i+1, total_models, clean_model_name, current_progress),
                current_progress
            )

            model_type, config_path, start_check_point = get_model_config(clean_model_name, auto_chunk_size, auto_overlap)

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
            last_yield_percent = -1
            downloading_file = None
            
            for line in process.stdout:
                line_stripped = line.strip()
                
                # Check for download progress [SESA_DOWNLOAD]
                if line_stripped.startswith("[SESA_DOWNLOAD]"):
                    try:
                        dl_info = line_stripped.replace("[SESA_DOWNLOAD]", "")
                        if dl_info.startswith("START:"):
                            downloading_file = dl_info.replace("START:", "")
                            yield None, f"📥 İndiriliyor: {downloading_file}", update_progress_html(
                                f"Model indiriliyor: {downloading_file}",
                                i * model_progress_per_step,
                                download_info={"filename": downloading_file, "percent": 0}
                            )
                        elif dl_info.startswith("END:"):
                            downloading_file = None
                        elif ":" in dl_info:
                            parts = dl_info.rsplit(":", 1)
                            if len(parts) == 2:
                                filename, percent_str = parts
                                download_percent = int(percent_str)
                                yield None, f"📥 İndiriliyor: {filename} - %{download_percent}", update_progress_html(
                                    f"Model indiriliyor: {filename}",
                                    i * model_progress_per_step,
                                    download_info={"filename": filename, "percent": download_percent}
                                )
                    except (ValueError, TypeError):
                        pass
                # Check for unique progress prefix [SESA_PROGRESS]
                elif line_stripped.startswith("[SESA_PROGRESS]"):
                    try:
                        # Extract percentage from [SESA_PROGRESS]XX format
                        percentage_str = line_stripped.replace("[SESA_PROGRESS]", "").strip()
                        percentage = float(percentage_str) if percentage_str else 0
                        percentage = min(max(percentage, 0), 100)  # Clamp to 0-100
                        
                        model_percentage = (percentage / 100) * model_progress_per_step
                        current_progress = (i * model_progress_per_step) + model_percentage
                        current_progress = clamp_percentage(current_progress)
                        
                        # Yield on every percent change for smooth updates
                        if int(percentage) != last_yield_percent:
                            last_yield_percent = int(percentage)
                            yield None, i18n("loading_model_progress_label").format(i+1, total_models, clean_model_name, int(percentage)), update_progress_html(
                                f"Model {i+1}/{total_models}: {clean_model_name} - {int(percentage)}%",
                                current_progress
                            )
                    except (ValueError, TypeError):
                        # Silently ignore parsing errors for progress lines
                        pass
                # Also support legacy "Progress: XX%" format for backwards compatibility
                elif line_stripped.startswith("Progress:"):
                    try:
                        match = re.search(r"Progress:\s*(\d+(?:\.\d+)?)%?", line_stripped)
                        if match:
                            percentage = float(match.group(1))
                            percentage = min(max(percentage, 0), 100)
                            
                            model_percentage = (percentage / 100) * model_progress_per_step
                            current_progress = (i * model_progress_per_step) + model_percentage
                            current_progress = clamp_percentage(current_progress)
                            
                            if int(percentage) != last_yield_percent:
                                last_yield_percent = int(percentage)
                                yield None, i18n("loading_model_progress_label").format(i+1, total_models, clean_model_name, int(percentage)), update_progress_html(
                                    f"Model {i+1}/{total_models}: {clean_model_name} - {int(percentage)}%",
                                    current_progress
                                )
                    except (ValueError, TypeError):
                        pass
                else:
                    # Print non-progress lines
                    if line_stripped:
                        print(line_stripped)

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
        else:
            drive_output_path = output_path

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
