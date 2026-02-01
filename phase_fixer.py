import torch
import torchaudio
import os
import gc

def frequency_blend_phases(phase1, phase2, freq_bins, low_cutoff=500, high_cutoff=5000, base_factor=0.25, scale_factor=1.85):
    if phase1.shape != phase2.shape:
        raise ValueError("phase1 and phase2 must have the same shape.")
    if len(freq_bins) != phase1.shape[0]:
        raise ValueError("freq_bins must have the same length as the number of frequency bins in phase1 and phase2.")
    if low_cutoff >= high_cutoff:
        raise ValueError("low_cutoff must be less than high_cutoff.")

    blended_phase = torch.zeros_like(phase1)
    blend_factors = torch.zeros_like(freq_bins)

    blend_factors[freq_bins < low_cutoff] = base_factor
    blend_factors[freq_bins > high_cutoff] = base_factor + scale_factor

    in_range_mask = (freq_bins >= low_cutoff) & (freq_bins <= high_cutoff)
    blend_factors[in_range_mask] = base_factor + scale_factor * (
        (freq_bins[in_range_mask] - low_cutoff) / (high_cutoff - low_cutoff)
    )

    for i in range(phase1.shape[0]):
        blended_phase[i, :] = (1 - blend_factors[i]) * phase1[i, :] + blend_factors[i] * phase2[i, :]

    blended_phase = torch.remainder(blended_phase + torch.pi, 2 * torch.pi) - torch.pi

    return blended_phase

def transfer_magnitude_phase(source_file, target_file, output_folder, transfer_magnitude=False, transfer_phase=True, 
                              low_cutoff=500, high_cutoff=9000, scale_factor=1.4, output_format='flac'):
    target_name, target_ext = os.path.splitext(os.path.basename(target_file))
    
    target_name = target_name.replace("_other", "").replace("_vocals", "").replace("_instrumental", "")
    target_name = target_name.replace("_Other", "").replace("_Vocals", "").replace("_Instrumental", "").strip()
    
    ext = '.flac' if output_format == 'flac' else '.wav'
    output_file = os.path.join(output_folder, f"{target_name} (Fixed Instrumental){ext}")

    print(f"Phase Fixing: {os.path.basename(target_file)}...")
    source_waveform, source_sr = torchaudio.load(source_file)
    target_waveform, target_sr = torchaudio.load(target_file)

    if source_sr != target_sr:
        raise ValueError("Sample rates of source and target audio files must match.")

    n_fft = 2048
    hop_length = 512
    window = torch.hann_window(n_fft)

    source_stfts = torch.stft(source_waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, pad_mode="reflect")
    target_stfts = torch.stft(target_waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, pad_mode="reflect")

    freqs = torch.linspace(0, source_sr // 2, steps=n_fft // 2 + 1)

    modified_stfts = []
    for source_stft, target_stft in zip(source_stfts, target_stfts):
        source_mag, source_phs = torch.abs(source_stft), torch.angle(source_stft)
        target_mag, target_phs = torch.abs(target_stft), torch.angle(target_stft)

        modified_stft = target_stft.clone()
        if transfer_magnitude:
            modified_stft = source_mag * torch.exp(1j * torch.angle(modified_stft))

        if transfer_phase:
            blended_phase = frequency_blend_phases(target_phs, source_phs, freqs, low_cutoff, high_cutoff, scale_factor=scale_factor)
            modified_stft = torch.abs(modified_stft) * torch.exp(1j * blended_phase)

        modified_stfts.append(modified_stft)

    modified_waveform = torch.istft(
        torch.stack(modified_stfts),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        length=source_waveform.size(1)
    )

    if output_format == 'flac':
        torchaudio.save(output_file, modified_waveform, target_sr, format="flac", bits_per_sample=16)
    else:
        torchaudio.save(output_file, modified_waveform, target_sr)
    
    print(f"Saved: {output_file}")
    return output_file

def process_phase_fix(source_file, target_file, output_folder, low_cutoff=500, high_cutoff=9000, 
                      scale_factor=1.4, output_format='flac'):
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        output_file = transfer_magnitude_phase(
            source_file=source_file,
            target_file=target_file,
            output_folder=output_folder,
            transfer_magnitude=False,
            transfer_phase=True,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
            scale_factor=scale_factor,
            output_format=output_format
        )
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return output_file, "Phase fix completed successfully!"
    except Exception as e:
        return None, f"Error during phase fix: {str(e)}"

SOURCE_MODELS = [
    'VOCALS-MelBand-Roformer (by Becruily)',
    'VOCALS-Mel-Roformer big beta 4 (by unwa)',
    'VOCALS-Melband-Roformer BigBeta5e (by unwa)',
    'VOCALS-big_beta6 (by Unwa)',
    'VOCALS-big_beta6X (by Unwa)',
    'VOCALS-MelBand-Roformer (by KimberleyJSN)',
    'VOCALS-MelBand-Roformer Kim FT (by Unwa)',
    'VOCALS-MelBand-Roformer Kim FT 2 (by Unwa)',
    'VOCALS-MelBand-Roformer Kim FT 2 Blendless (by unwa)',
    'VOCALS-Mel-Roformer FT 3 Preview (by unwa)',
    'VOCALS-BS-Roformer_1296 (by viperx)',
    'VOCALS-BS-Roformer_1297 (by viperx)',
    'VOCALS-BS-RoformerLargev1 (by unwa)',
    'bs_roformer_revive (by unwa)'
]

TARGET_MODELS = [
    'INST-MelBand-Roformer (by Becruily)',
    'INST-Mel-Roformer v1 (by unwa)',
    'INST-Mel-Roformer v2 (by unwa)',
    'inst_v1e (by unwa)',
    'INST-Mel-Roformer v1e+ (by unwa)',
    'Inst_GaboxV7 (by Gabox)',
    'INST-VOC-Mel-Roformer a.k.a. duality (by unwa)',
    'INST-VOC-Mel-Roformer a.k.a. duality v2 (by unwa)',
    'inst_gabox (by Gabox)',
    'inst_gaboxFlowersV10 (by Gabox)'
]
