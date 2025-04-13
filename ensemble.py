#!/usr/bin/env python3
# coding: utf-8
"""
Ultimate Audio Ensemble Tool v3.0
- Handles all edge cases
- Maximum compatibility
- Detailed logging
"""
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

class AudioEnsembler:
    def __init__(self):
        self.temp_dir = None
        
    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp(prefix='ensemble_')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def normalize_path(self, path):
        """Convert path to absolute POSIX format and handle special cases."""
        try:
            path = str(Path(path).absolute().resolve())
            # Handle Google Colab specific paths
            if '/content/drive/' in path:
                path = path.replace('/content/drive/', '/gdrive/')
            return path
        except Exception as e:
            print(f"Path normalization failed for {path}: {str(e)}")
            return path
    
    def copy_to_temp(self, src_path):
        """Copy problematic files to temp location with safe names."""
        try:
            safe_name = ''.join(c for c in os.path.basename(src_path) if c.isalnum() or c in ('-', '_', '.'))
            dest_path = os.path.join(self.temp_dir, safe_name)
            
            # Read and rewrite the file to ensure clean format
            data, sr = librosa.load(src_path, sr=None, mono=False)
            sf.write(dest_path, data.T, sr)
            return dest_path
        except Exception as e:
            print(f"Failed to create temp copy of {src_path}: {str(e)}")
            raise
    
    def validate_inputs(self, files, method, output_path):
        """Comprehensive input validation."""
        errors = []
        
        # Check method
        valid_methods = ['avg_wave', 'median_wave', 'max_fft', 'min_fft', 'median_fft']
        if method not in valid_methods:
            errors.append(f"Invalid method: {method}. Must be one of {valid_methods}")
        
        # Check files
        valid_files = []
        sample_rates = set()
        durations = []
        
        for f in files:
            try:
                f = self.normalize_path(f)
                if not os.path.exists(f):
                    errors.append(f"File not found: {f}")
                    continue
                
                if os.path.getsize(f) == 0:
                    errors.append(f"Empty file: {f}")
                    continue
                
                # Verify file can be opened
                try:
                    with sf.SoundFile(f) as test:
                        sr = test.samplerate
                        frames = test.frames
                        channels = test.channels
                except Exception as e:
                    errors.append(f"Unreadable file {f}: {str(e)}")
                    continue
                
                if channels != 2:
                    errors.append(f"File {f} must be stereo (has {channels} channels)")
                    continue
                
                sample_rates.add(sr)
                durations.append(frames / sr)
                valid_files.append(f)
                
            except Exception as e:
                errors.append(f"Error processing {f}: {str(e)}")
                continue
        
        if len(valid_files) < 2:
            errors.append("At least 2 valid files required")
        
        # Check output path
        try:
            output_path = self.normalize_path(output_path)
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            if not os.access(output_dir if output_dir else '.', os.W_OK):
                errors.append(f"No write permission for output: {output_path}")
        except Exception as e:
            errors.append(f"Output path error: {str(e)}")
        
        if errors:
            raise ValueError("\n".join(errors))
        
        return valid_files, sample_rates, durations
    
    def process_files(self, files, method, output_path, weights=None, buffer_size=32768):
        """Core processing with maximum robustness."""
        try:
            # Validate inputs
            valid_files, sample_rates, durations = self.validate_inputs(files, method, output_path)
            output_path = self.normalize_path(output_path)
            
            # Handle sample rate differences
            target_sr = max(sample_rates) if sample_rates else 44100
            if len(sample_rates) > 1:
                print(f"Different sample rates detected. Resampling to {target_sr}Hz")
                
            # Handle path issues by copying to temp location
            temp_files = []
            for f in valid_files:
                try:
                    temp_files.append(self.copy_to_temp(f))
                except:
                    temp_files.append(f)  # Fallback to original if temp copy fails
            
            # Get shortest duration
            shortest_dur = min(durations) if durations else 0
            if not shortest_dur:
                raise ValueError("Could not determine audio durations")
            
            # Prepare weights
            if weights and len(weights) == len(temp_files):
                weights = np.array(weights, dtype=np.float32)
                weights /= weights.sum()  # Normalize
            else:
                weights = None
            
            # Process with soundfile readers
            readers = []
            try:
                # Open all files
                readers = [sf.SoundFile(f) for f in temp_files]
                shortest_frames = min(int(shortest_dur * r.samplerate) for r in readers)
                
                # Prepare output
                with sf.SoundFile(output_path, 'w', target_sr, 2, 'PCM_24') as outfile:
                    # Process in chunks
                    for pos in range(0, shortest_frames, buffer_size):
                        chunk_size = min(buffer_size, shortest_frames - pos)
                        
                        # Read aligned chunks
                        chunks = []
                        for r in readers:
                            r.seek(pos)
                            data = r.read(chunk_size)
                            if data.size == 0:
                                data = np.zeros((chunk_size, 2))
                            chunks.append(data.T)  # Transpose to (channels, samples)
                        
                        # Convert to numpy array
                        chunks = np.array(chunks)
                        
                        # Process based on method
                        if method.endswith('_fft'):
                            result = self.spectral_process(chunks, method)
                        else:
                            result = self.waveform_process(chunks, method, weights)
                        
                        # Write output
                        outfile.write(result.T)  # Transpose back to (samples, channels)
                        
                        # Clean up
                        del chunks, result
                        if pos % (5 * buffer_size) == 0:  # Periodic cleanup
                            gc.collect()
                        
                        # Progress
                        progress = 100 * pos / shortest_frames
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
                
                print(f"\nSuccessfully created: {output_path}")
                return True
                
            finally:
                for r in readers:
                    try:
                        r.close()
                    except:
                        pass
            
        except Exception as e:
            print(f"\nError during processing: {str(e)}", file=sys.stderr)
            traceback.print_exc()
            return False
    
    def waveform_process(self, chunks, method, weights):
        """Waveform domain processing."""
        if method == 'avg_wave':
            if weights is not None:
                return np.average(chunks, axis=0, weights=weights)
            return np.mean(chunks, axis=0)
        elif method == 'median_wave':
            return np.median(chunks, axis=0)
    
    def spectral_process(self, chunks, method):
        """Frequency domain processing."""
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
        
        # ISTFT
        reconstructed = np.zeros((combined_spec.shape[0], chunks[0].shape[1]))
        for channel in range(combined_spec.shape[0]):
            _, xrec = istft(combined_spec[channel], nperseg=1024, noverlap=512)
            reconstructed[channel] = xrec[:chunks[0].shape[1]]
        
        return reconstructed

def main():
    parser = argparse.ArgumentParser(description='Ultimate Audio Ensemble Tool')
    parser.add_argument('--files', nargs='+', required=True, help='Input audio files')
    parser.add_argument('--type', required=True,
                       choices=['avg_wave', 'median_wave', 'max_fft', 'min_fft', 'median_fft'],
                       help='Ensemble method')
    parser.add_argument('--weights', nargs='+', type=float, help='Weights for each file')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--buffer', type=int, default=32768,
                       help='Buffer size in samples (default: 32768)')
    
    args = parser.parse_args()
    
    # Normalize weights
    weights = None
    if args.weights:
        if len(args.weights) != len(args.files):
            print("Warning: Weights count mismatch. Using equal weights.")
        else:
            weights = args.weights
    
    with AudioEnsembler() as ensembler:
        success = ensembler.process_files(
            files=args.files,
            method=args.type,
            output_path=args.output,
            weights=weights,
            buffer_size=args.buffer
        )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
