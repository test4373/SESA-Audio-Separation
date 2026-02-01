# Kaggle için gerekli kütüphaneleri yükleme
!pip install librosa soundfile psutil tqdm
import os
import sys
import argparse
import numpy as np
import soundfile as sf
import librosa
import psutil
import gc
import traceback
from scipy.signal import stft, istft
from pathlib import Path
import tempfile
import shutil
import json
from tqdm import tqdm
import time

class AudioEnsembleEngine:
    def __init__(self):
        self.temp_dir = None
        self.log_file = "/kaggle/working/ensemble_processor.log"
        
    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp(prefix='audio_ensemble_', dir='/kaggle/working/')
        self.setup_logging()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def setup_logging(self):
        """Initialize detailed logging system."""
        with open(self.log_file, 'w') as f:
            f.write("Audio Ensemble Processor Log\n")
            f.write("="*50 + "\n")
            f.write(f"System Memory: {psutil.virtual_memory().total/(1024**3):.2f} GB\n")
            f.write(f"Python Version: {sys.version}\n\n")
    
    def log_message(self, message):
        """Log messages with timestamp."""
        with open(self.log_file, 'a') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    
    def normalize_path(self, path):
        """Handle all path-related issues comprehensively."""
        try:
            # Kaggle'da dosya yolları /kaggle/input/ veya /kaggle/working/ altında
            path = str(Path(path).absolute().resolve())
            
            # Handle problematic characters
            if any(char in path for char in '[]()|&; '):
                base, ext = os.path.splitext(path)
                safe_name = f"{hash(base)}{ext}"
                temp_path = os.path.join(self.temp_dir, safe_name)
                
                if not os.path.exists(temp_path):
                    data, sr = librosa.load(path, sr=None, mono=False)
                    sf.write(temp_path, data.T, sr)
                
                return temp_path
            
            return path
        except Exception as e:
            self.log_message(f"Path normalization failed: {str(e)}")
            return path
    
    def validate_inputs(self, files, method, output_path):
        """Comprehensive input validation with detailed error reporting."""
        errors = []
        valid_methods = [
            'avg_wave', 'median_wave', 'max_wave', 'min_wave',
            'max_fft', 'min_fft', 'median_fft'
        ]
        
        # Method validation
        if method not in valid_methods:
            errors.append(f"Invalid method '{method}'. Available: {valid_methods}")
        
        # File validation
        valid_files = []
        sample_rates = set()
        durations = []
        channels_set = set()
        
        for f in files:
            try:
                f_normalized = self.normalize_path(f)
                
                # Basic checks
                if not os.path.exists(f_normalized):
                    errors.append(f"File not found: {f_normalized}")
                    continue
                
                if os.path.getsize(f_normalized) == 0:
                    errors.append(f"Empty file: {f_normalized}")
                    continue
                
                # Audio file validation
                try:
                    with sf.SoundFile(f_normalized) as sf_file:
                        sr = sf_file.samplerate
                        frames = sf_file.frames
                        channels = sf_file.channels
                except Exception as e:
                    errors.append(f"Invalid audio file {f_normalized}: {str(e)}")
                    continue
                
                # Audio characteristics
                if channels != 2:
                    errors.append(f"File must be stereo (has {channels} channels): {f_normalized}")
                    continue
                
                sample_rates.add(sr)
                durations.append(frames / sr)
                channels_set.add(channels)
                valid_files.append(f_normalized)
                
            except Exception as e:
                errors.append(f"Error processing {f}: {str(e)}")
                continue
        
        # Final checks
        if len(valid_files) < 2:
            errors.append("At least 2 valid files required")
        
        if len(sample_rates) > 1:
            errors.append(f"Sample rate mismatch: {sample_rates}")
        
        if len(channels_set) > 1:
            errors.append(f"Channel count mismatch: {channels_set}")
        
        # Output path validation
        try:
            output_path = self.normalize_path(output_path)
            output_dir = os.path.dirname(output_path) or '/kaggle/working/'
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            if not os.access(output_dir, os.W_OK):
                errors.append(f"No write permission for output directory: {output_dir}")
        except Exception as e:
            errors.append(f"Output path error: {str(e)}")
        
        if errors:
            error_msg = "\n".join(errors)
            self.log_message(f"Validation failed:\n{error_msg}")
            raise ValueError(error_msg)
        
        target_sr = sample_rates.pop() if sample_rates else 44100
        return valid_files, target_sr, min(durations) if durations else None
    
    def process_waveform(self, chunks, method, weights=None):
        """All waveform domain processing methods."""
        if method == 'avg_wave':
            if weights is not None:
                return np.average(chunks, axis=0, weights=weights)
            return np.mean(chunks, axis=0)
        elif method == 'median_wave':
            return np.median(chunks, axis=0)
        elif method == 'max_wave':
            return np.max(chunks, axis=0)
        elif method == 'min_wave':
            return np.min(chunks, axis=0)
    
    def process_spectral(self, chunks, method):
        """All frequency domain processing methods."""
        specs = []
        for c in chunks:
            channel_specs = []
            for channel in range(c.shape[0]):
                _, _, Zxx = stft(c[channel], nperseg=1024, noverlap=512)
                channel_specs.append(Zxx)
            specs.append(np.array(channel_specs))
        
        specs = np.array(specs)
        mag = np.abs(specs)
        
        if method == 'max_fft':
            combined_mag = np.max(mag, axis=0)
        elif method == 'min_fft':
            combined_mag = np.min(mag, axis=0)
        elif method == 'median_fft':
            combined_mag = np.median(mag, axis=0)
        
        # Use phase from first file
        combined_spec = combined_mag * np.exp(1j * np.angle(specs[0]))
        
        # ISTFT reconstruction
        reconstructed = np.zeros((combined_spec.shape[0], chunks[0].shape[1]))
        for channel in range(combined_spec.shape[0]):
            _, xrec = istft(combined_spec[channel], nperseg=1024, noverlap=512)
            reconstructed[channel] = xrec[:chunks[0].shape[1]]
        
        return reconstructed
    
    def run_ensemble(self, files, method, output_path, weights=None, buffer_size=32768):
        """Core ensemble processing with maximum robustness."""
        try:
            # Validate and prepare inputs
            valid_files, target_sr, duration = self.validate_inputs(files, method, output_path)
            output_path = self.normalize_path(output_path)
            
            self.log_message(f"Starting ensemble with method: {method}")
            self.log_message(f"Input files: {json.dumps(valid_files, indent=2)}")
            self.log_message(f"Target sample rate: {target_sr}Hz")
            self.log_message(f"Output path: {output_path}")
            
            # Prepare weights
            if weights and len(weights) == len(valid_files):
                weights = np.array(weights, dtype=np.float32)
                weights /= weights.sum()  # Normalize
                self.log_message(f"Using weights: {weights}")
            else:
                weights = None
            
            # Open all files
            readers = []
            try:
                readers = [sf.SoundFile(f) for f in valid_files]
                shortest_frames = min(int(duration * r.samplerate) for r in readers)
                
                # Prepare output
                with sf.SoundFile(output_path, 'w', target_sr, 2, 'PCM_24') as outfile:
                    # Process in chunks with progress bar
                    progress = tqdm(total=shortest_frames, unit='samples', desc='Processing')
                    
                    for pos in range(0, shortest_frames, buffer_size):
                        chunk_size = min(buffer_size, shortest_frames - pos)
                        
                        # Read aligned chunks from all files
                        chunks = []
                        for r in readers:
                            r.seek(pos)
                            data = r.read(chunk_size)
                            if data.size == 0:
                                data = np.zeros((chunk_size, 2))
                            chunks.append(data.T)  # Transpose to (channels, samples)
                        
                        chunks = np.array(chunks)
                        
                        # Process based on method type
                        if method.endswith('_fft'):
                            result = self.process_spectral(chunks, method)
                        else:
                            result = self.process_waveform(chunks, method, weights)
                        
                        # Write output
                        outfile.write(result.T)  # Transpose back to (samples, channels)
                        
                        # Clean up and update progress
                        del chunks, result
                        if pos % (5 * buffer_size) == 0:
                            gc.collect()
                        
                        progress.update(chunk_size)
                    
                    progress.close()
                
                self.log_message(f"Successfully created output: {output_path}")
                print(f"\nEnsemble completed successfully: {output_path}")
                return True
                
            except Exception as e:
                self.log_message(f"Processing error: {str(e)}\n{traceback.format_exc()}")
                raise
            finally:
                for r in readers:
                    try:
                        r.close()
                    except:
                        pass
                
        except Exception as e:
            self.log_message(f"Fatal error: {str(e)}\n{traceback.format_exc()}")
            print(f"\nError during processing: {str(e)}", file=sys.stderr)
            return False

def main():
    parser = argparse.ArgumentParser(
        description='Ultimate Audio Ensemble Processor - Supports all ensemble methods',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--files', nargs='+', required=True,
                       help='Input audio files (supports special characters)')
    parser.add_argument('--type', required=True,
                       choices=['avg_wave', 'median_wave', 'max_wave', 'min_wave',
                               'max_fft', 'min_fft', 'median_fft'],
                       help='Ensemble method to use')
    parser.add_argument('--weights', nargs='+', type=float,
                       help='Relative weights for each input file')
    parser.add_argument('--output', required=True,
                       help='Output file path')
    parser.add_argument('--buffer', type=int, default=32768,
                       help='Buffer size in samples (larger=faster but uses more memory)')
    
    args = parser.parse_args()
    
    with AudioEnsembleEngine() as engine:
        success = engine.run_ensemble(
            files=args.files,
            method=args.type,
            output_path=args.output,
            weights=args.weights,
            buffer_size=args.buffer
        )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
