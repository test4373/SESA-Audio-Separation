# apollo_processing.py
import os
import subprocess
import librosa
import soundfile as sf
import numpy as np
from helpers import clamp_percentage, sanitize_filename

def process_with_apollo(
    output_files,
    output_dir,
    apollo_chunk_size,
    apollo_overlap,
    apollo_method,
    apollo_normal_model,
    apollo_midside_model,
    output_format,
    progress=None,
    total_progress_start=80,
    total_progress_end=100
):
    """
    Process audio files with Apollo enhancement.
    
    Args:
        output_files: List of input audio file paths to process.
        output_dir: Directory to store enhanced output files.
        apollo_chunk_size: Chunk size for Apollo processing.
        apollo_overlap: Overlap for Apollo processing.
        apollo_method: Apollo processing method ('normal_method' or 'mid_side_method').
        apollo_normal_model: Apollo model for normal method.
        apollo_midside_model: Apollo model for mid-side method.
        output_format: Output audio format (e.g., 'wav').
        progress: Gradio progress object for UI updates.
        total_progress_start: Starting progress percentage (default: 80).
        total_progress_end: Ending progress percentage (default: 100).

    Returns:
        List of enhanced file paths or original files if processing fails.
    """
    try:
        apollo_script = "/content/Apollo/inference.py"
        print(f"Apollo parameters - chunk_size: {apollo_chunk_size}, overlap: {apollo_overlap}, method: {apollo_method}, normal_model: {apollo_normal_model}, midside_model: {apollo_midside_model}")

        # Select checkpoint and config based on method and model
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
            raise FileNotFoundError(f"Apollo configuration file not found: {config}")

        enhanced_files = []
        total_files = len([f for f in output_files if f and os.path.exists(f)])
        progress_per_file = (total_progress_end - total_progress_start) / total_files if total_files > 0 else (total_progress_end - total_progress_start)

        for idx, output_file in enumerate(output_files):
            if output_file and os.path.exists(output_file):
                original_file_name = sanitize_filename(os.path.splitext(os.path.basename(output_file))[0])
                enhancement_suffix = "_Mid_Side_Enhanced" if apollo_method == "mid_side_method" else "_Enhanced"
                enhanced_output = os.path.join(output_dir, f"{original_file_name}{enhancement_suffix}.{output_format}")

                try:
                    # Progress update
                    if progress is not None and callable(getattr(progress, '__call__', None)):
                        current_progress = total_progress_start + (idx * progress_per_file)
                        current_progress = clamp_percentage(current_progress)
                        progress(current_progress, desc=f"Enhancing with Apollo... ({idx+1}/{total_files})")
                    else:
                        print(f"Progress is not callable or None, skipping Apollo progress update: file {idx+1}/{total_files}")

                    if apollo_method == "mid_side_method":
                        audio, sr = librosa.load(output_file, mono=False, sr=None)
                        if audio.ndim == 1:
                            audio = np.array([audio, audio])

                        mid = (audio[0] + audio[1]) * 0.5
                        side = (audio[0] - audio[1]) * 0.5

                        mid_file = os.path.join(output_dir, f"{original_file_name}_mid_temp.wav")
                        side_file = os.path.join(output_dir, f"{original_file_name}_side_temp.wav")
                        sf.write(mid_file, mid, sr)
                        sf.write(side_file, side, sr)

                        mid_output = os.path.join(output_dir, f"{original_file_name}_mid_enhanced.{output_format}")
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

                        side_output = os.path.join(output_dir, f"{original_file_name}_side_enhanced.{output_format}")
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
                                print(f"Could not delete temporary file {temp_file}: {str(e)}")

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
                            print(f"Apollo processing failed: {output_file}: {apollo_process.stderr}")
                            enhanced_files.append(output_file)
                            continue

                        if not os.path.exists(enhanced_output):
                            print(f"Apollo output missing: {enhanced_output}")
                            enhanced_files.append(output_file)
                            continue

                        enhanced_files.append(enhanced_output)

                    # Progress update after each file
                    if progress is not None and callable(getattr(progress, '__call__', None)):
                        current_progress = total_progress_start + ((idx + 1) * progress_per_file)
                        current_progress = clamp_percentage(current_progress)
                        progress(current_progress, desc=f"Enhancing with Apollo... ({idx+1}/{total_files})")

                except Exception as e:
                    print(f"Error during Apollo processing: {output_file}: {str(e)}")
                    enhanced_files.append(output_file)
                    continue
            else:
                enhanced_files.append(output_file)

        # Final progress update
        if progress is not None and callable(getattr(progress, '__call__', None)):
            progress(total_progress_end, desc="Apollo enhancement complete")

        return enhanced_files

    except Exception as e:
        print(f"Apollo processing error: {str(e)}")
        return [f for f in output_files]