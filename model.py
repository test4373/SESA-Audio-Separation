import os
import yaml
from urllib.parse import quote
from pathlib import Path

# Temel dizin ve checkpoint dizini sabit olarak tanımlanıyor
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'ckpts')

def conf_edit(config_path, chunk_size, overlap):
    """Edits the configuration file with chunk size and overlap."""
    full_config_path = os.path.join(CHECKPOINT_DIR, os.path.basename(config_path))
    if not os.path.exists(full_config_path):
        raise FileNotFoundError(f"Configuration file not found: {full_config_path}")
    
    with open(full_config_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    if 'use_amp' not in data.keys():
        data['training']['use_amp'] = True

    data['audio']['chunk_size'] = chunk_size
    data['inference']['num_overlap'] = overlap
    if data['inference']['batch_size'] == 1:
        data['inference']['batch_size'] = 2

    print(f"Using custom overlap and chunk_size: overlap={overlap}, chunk_size={chunk_size}")
    with open(full_config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, Dumper=yaml.Dumper)

def download_file(url):
    """Downloads a file from a URL."""
    import requests
    encoded_url = quote(url, safe=':/')
    path = CHECKPOINT_DIR
    os.makedirs(path, exist_ok=True)
    filename = os.path.basename(encoded_url)
    file_path = os.path.join(path, filename)
    if os.path.exists(file_path):
        print(f"File '{filename}' already exists at '{path}'.")
        return
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"File '{filename}' downloaded successfully")
        else:
            print(f"Error downloading '{filename}': Status code {response.status_code}")
    except Exception as e:
        print(f"Error downloading file '{filename}' from '{url}': {e}")

# Model konfigurasyonlarını kategorize bir sözlükte tut
MODEL_CONFIGS = {
    "Vocal Models": {
        'VOCALS-InstVocHQ': {
            'model_type': 'mdx23c',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_vocals_mdx23c.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model_vocals_mdx23c_sdr_10.17.ckpt'),
            'download_urls': [
                'https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_vocals_mdx23c.yaml',
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt'
            ],
            'needs_conf_edit': False
        },
        'VOCALS-MelBand-Roformer (by KimberleyJSN)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_vocals_mel_band_roformer_kj.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'MelBandRoformer.ckpt'),
            'download_urls': [
                'https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml',
                'https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt'
            ],
            'needs_conf_edit': True
        },
        'VOCALS-BS-Roformer_1297 (by viperx)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'model_bs_roformer_ep_317_sdr_12.9755.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model_bs_roformer_ep_317_sdr_12.9755.ckpt'),
            'download_urls': [
                'https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml',
                'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt'
            ],
            'needs_conf_edit': True
        },
        'VOCALS-BS-Roformer_1296 (by viperx)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'model_bs_roformer_ep_368_sdr_12.9628.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model_bs_roformer_ep_368_sdr_12.9628.ckpt'),
            'download_urls': [
                'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt',
                'https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_368_sdr_12.9628.yaml'
            ],
            'needs_conf_edit': True
        },
        'VOCALS-BS-RoformerLargev1 (by unwa)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_bsrofoL.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'BS-Roformer_LargeV1.ckpt'),
            'download_urls': [
                'https://huggingface.co/jarredou/unwa_bs_roformer/resolve/main/BS-Roformer_LargeV1.ckpt',
                'https://huggingface.co/jarredou/unwa_bs_roformer/raw/main/config_bsrofoL.yaml'
            ],
            'needs_conf_edit': True
        },
        'VOCALS-Mel-Roformer big beta 4 (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_melbandroformer_big_beta4.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'melband_roformer_big_beta4.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/melband_roformer_big_beta4.ckpt',
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-big/raw/main/config_melbandroformer_big_beta4.yaml'
            ],
            'needs_conf_edit': True
        },
        'VOCALS-Melband-Roformer BigBeta5e (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'big_beta5e.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'big_beta5e.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.ckpt',
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-big/resolve/main/big_beta5e.yaml'
            ],
            'needs_conf_edit': True
        },
        'VOCALS-VitLarge23 (by ZFTurbo)': {
            'model_type': 'segm_models',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_vocals_segm_models.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model_vocals_segm_models_sdr_9.77.ckpt'),
            'download_urls': [
                'https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/refs/heads/main/configs/config_vocals_segm_models.yaml',
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_segm_models_sdr_9.77.ckpt'
            ],
            'needs_conf_edit': False
        },
        'VOCALS-MelBand-Roformer Kim FT (by Unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_kimmel_unwa_ft.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'kimmel_unwa_ft.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft.ckpt',
                'https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml'
            ],
            'needs_conf_edit': True
        },
        'VOCALS-MelBand-Roformer (by Becruily)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_instrumental_becruily.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'mel_band_roformer_vocals_becruily.ckpt'),
            'download_urls': [
                'https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/config_vocals_becruily.yaml',
                'https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/mel_band_roformer_vocals_becruily.ckpt'
            ],
            'needs_conf_edit': True
        },
        'VOCALS-Male Female-BS-RoFormer Male Female Beta 7_2889 (by aufr33)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_chorus_male_female_bs_roformer.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt'),
            'download_urls': [
                'https://huggingface.co/RareSirMix/AIModelRehosting/resolve/main/bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt',
                'https://huggingface.co/Sucial/Chorus_Male_Female_BS_Roformer/resolve/main/config_chorus_male_female_bs_roformer.yaml'
            ],
            'needs_conf_edit': True
        },
        'VOCALS-MelBand-Roformer Kim FT 2 (by Unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_kimmel_unwa_ft.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'kimmel_unwa_ft2.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml',
                'https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft2.ckpt'
            ],
            'needs_conf_edit': True
        },
        'voc_gaboxBSroformer (by Gabox)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'voc_gaboxBSroformer.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'voc_gaboxBSR.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/BSRoformerVocTest/resolve/main/voc_gaboxBSroformer.yaml',
                'https://huggingface.co/GaboxR67/BSRoformerVocTest/resolve/main/voc_gaboxBSR.ckpt'
            ],
            'needs_conf_edit': True
        },
        'voc_gaboxMelReformer (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'voc_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'voc_gabox.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.ckpt'
            ],
            'needs_conf_edit': True
        },
        'voc_gaboxMelReformerFV1 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'voc_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'voc_gaboxFv1.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gaboxFv1.ckpt'
            ],
            'needs_conf_edit': True
        },
        'voc_gaboxMelReformerFV2 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'voc_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'voc_gaboxFv2.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gaboxFv2.ckpt'
            ],
            'needs_conf_edit': True
        },
        'VOCALS-MelBand-Roformer Kim FT 2 Blendless (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_kimmel_unwa_ft.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'kimmel_unwa_ft2_bleedless.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml',
                'https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft2_bleedless.ckpt'
            ],
            'needs_conf_edit': True
        },
        'Voc_Fv3 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'voc_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'voc_Fv3.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/vocals/voc_Fv3.ckpt'
            ],
            'needs_conf_edit': True
        },
        'FullnessVocalModel (by Amane)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'FullnessVocalModel.ckpt'),
            'download_urls': [
                'https://huggingface.co/Aname-Tommy/MelBandRoformers/blob/main/config.yaml',
                'https://huggingface.co/Aname-Tommy/MelBandRoformers/blob/main/FullnessVocalModel.ckpt'
            ],
            'needs_conf_edit': True
        }
    },
    "Instrumental Models": {
        'INST-Mel-Roformer v1 (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_melbandroformer_inst.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'melband_roformer_inst_v1.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v1.ckpt',
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst.yaml'
            ],
            'needs_conf_edit': True
        },
        'INST-Mel-Roformer v1e+ (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_melbandroformer_inst.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'melband_roformer_inst_v1.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e_plus.ckpt',
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst.yaml'
            ],
            'needs_conf_edit': True
        },
        'INST-Mel-Roformer v1+ (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_melbandroformer_inst.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'melband_roformer_inst_v1.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1_plus_test.ckpt',
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst.yaml'
            ],
            'needs_conf_edit': True
        },
        'INST-Mel-Roformer v2 (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_melbandroformer_inst_v2.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'melband_roformer_inst_v2.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/melband_roformer_inst_v2.ckpt',
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/raw/main/config_melbandroformer_inst_v2.yaml'
            ],
            'needs_conf_edit': True
        },
        'INST-VOC-Mel-Roformer a.k.a. duality (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_melbandroformer_instvoc_duality.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'melband_roformer_instvoc_duality_v1.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvoc_duality_v1.ckpt',
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/raw/main/config_melbandroformer_instvoc_duality.yaml'
            ],
            'needs_conf_edit': True
        },
        'INST-VOC-Mel-Roformer a.k.a. duality v2 (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_melbandroformer_instvoc_duality.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'melband_roformer_instvox_duality_v2.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/resolve/main/melband_roformer_instvox_duality_v2.ckpt',
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-InstVoc-Duality/raw/main/config_melbandroformer_instvoc_duality.yaml'
            ],
            'needs_conf_edit': True
        },
        'INST-MelBand-Roformer (by Becruily)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_instrumental_becruily.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'mel_band_roformer_instrumental_becruily.ckpt'),
            'download_urls': [
                'https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/config_instrumental_becruily.yaml',
                'https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt'
            ],
            'needs_conf_edit': True
        },
        'inst_v1e (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_melbandroformer_inst.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'inst_v1e.ckpt'),
            'download_urls': [
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e.ckpt',
                'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml'
            ],
            'needs_conf_edit': True
        },
        'inst_gabox (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'inst_gabox.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.ckpt'
            ],
            'needs_conf_edit': True
        },
        'inst_gaboxBV1 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'inst_gaboxBv1.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxBv1.ckpt'
            ],
            'needs_conf_edit': True
        },
        'inst_gaboxBV2 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'inst_gaboxBv2.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxBv2.ckpt'
            ],
            'needs_conf_edit': True
        },
        'inst_gaboxBFV1 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'gaboxFv1.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv1.ckpt'
            ],
            'needs_conf_edit': True
        },
        'inst_gaboxFV2 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'inst_gaboxFv2.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv2.ckpt'
            ],
            'needs_conf_edit': True
        },
        'inst_Fv3 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'inst_gaboxFv3.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv3.ckpt'
            ],
            'needs_conf_edit': True
        },
        'Intrumental_Gabox (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'intrumental_gabox.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/intrumental_gabox.ckpt'
            ],
            'needs_conf_edit': True
        },
        'inst_Fv4Noise (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'inst_Fv4Noise.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_Fv4Noise.ckpt'
            ],
            'needs_conf_edit': True
        },
        'INSTV5 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'INSTV5.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV5.ckpt'
            ],
            'needs_conf_edit': True
        },
        'inst_gaboxFV1 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'inst_gaboxFv1.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gaboxFv1.ckpt'
            ],
            'needs_conf_edit': True
        },
        'inst_gaboxFV6 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'INSTV6.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV6.ckpt'
            ],
            'needs_conf_edit': True
        },
        'INSTV5N (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'INSTV5N.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV5N.ckpt'
            ],
            'needs_conf_edit': True
        },
        'INSTV6N (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'INSTV6N.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/INSTV6N.ckpt'
            ],
            'needs_conf_edit': True
        },
        'Inst_GaboxV7 (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'inst_gabox.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'Inst_GaboxV7.ckpt'),
            'download_urls': [
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_gabox.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxV7.ckpt'
            ],
            'needs_conf_edit': True
        }
    },
    "4-Stem Models": {
        '4STEMS-SCNet_MUSDB18 (by starrytong)': {
            'model_type': 'scnet',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_musdb18_scnet.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'scnet_checkpoint_musdb18.ckpt'),
            'download_urls': [
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/config_musdb18_scnet.yaml',
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/scnet_checkpoint_musdb18.ckpt'
            ],
            'needs_conf_edit': False
        },
        '4STEMS-SCNet_XL_MUSDB18 (by ZFTurbo)': {
            'model_type': 'scnet',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_musdb18_scnet_xl.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model_scnet_ep_54_sdr_9.8051.ckpt'),
            'download_urls': [
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/config_musdb18_scnet_xl.yaml',
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/model_scnet_ep_54_sdr_9.8051.ckpt'
            ],
            'needs_conf_edit': True
        },
        '4STEMS-SCNet_Large (by starrytong)': {
            'model_type': 'scnet',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_musdb18_scnet_large_starrytong.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'SCNet-large_starrytong_fixed.ckpt'),
            'download_urls': [
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/config_musdb18_scnet_large_starrytong.yaml',
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.9/SCNet-large_starrytong_fixed.ckpt'
            ],
            'needs_conf_edit': True
        },
        '4STEMS-BS-Roformer_MUSDB18 (by ZFTurbo)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_bs_roformer_384_8_2_485100.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model_bs_roformer_ep_17_sdr_9.6568.ckpt'),
            'download_urls': [
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/config_bs_roformer_384_8_2_485100.yaml',
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt'
            ],
            'needs_conf_edit': True
        },
        'MelBandRoformer4StemFTLarge (SYH99999)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'MelBandRoformer4StemFTLarge.ckpt'),
            'download_urls': [
                'https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/config.yaml',
                'https://huggingface.co/SYH99999/MelBandRoformer4StemFTLarge/resolve/main/MelBandRoformer4StemFTLarge.ckpt'
            ],
            'needs_conf_edit': True
        }
    },
    "Denoise Models": {
        'DENOISE-MelBand-Roformer-1 (by aufr33)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'model_mel_band_roformer_denoise.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt'),
            'download_urls': [
                'https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt',
                'https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml'
            ],
            'needs_conf_edit': True
        },
        'DENOISE-MelBand-Roformer-2 (by aufr33)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'model_mel_band_roformer_denoise.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt'),
            'download_urls': [
                'https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt',
                'https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml'
            ],
            'needs_conf_edit': True
        },
        'denoisedebleed (by Gabox)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'model_mel_band_roformer_denoise.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'denoisedebleed.ckpt'),
            'download_urls': [
                'https://huggingface.co/poiqazwsx/melband-roformer-denoise/resolve/main/model_mel_band_roformer_denoise.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/denoisedebleed.ckpt'
            ],
            'needs_conf_edit': True
        }
    },
    "Dereverb Models": {
        'DE-REVERB-MDX23C (by aufr33 & jarredou)': {
            'model_type': 'mdx23c',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_dereverb_mdx23c.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'dereverb_mdx23c_sdr_6.9096.ckpt'),
            'download_urls': [
                'https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/dereverb_mdx23c_sdr_6.9096.ckpt',
                'https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/config_dereverb_mdx23c.yaml'
            ],
            'needs_conf_edit': False
        },
        'DE-REVERB-MelBand-Roformer aggr./v2/19.1729 (by anvuew)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'dereverb_mel_band_roformer_anvuew.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt'),
            'download_urls': [
                'https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt',
                'https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml'
            ],
            'needs_conf_edit': True
        },
        'DE-REVERB-Echo-MelBand-Roformer (by Sucial)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_dereverb-echo_mel_band_roformer.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt'),
            'download_urls': [
                'https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt',
                'https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer/resolve/main/config_dereverb-echo_mel_band_roformer.yaml'
            ],
            'needs_conf_edit': True
        },
        'dereverb_mel_band_roformer_less_aggressive_anvuew': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'dereverb_mel_band_roformer_anvuew.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt'),
            'download_urls': [
                'https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml',
                'https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt'
            ],
            'needs_conf_edit': True
        },
        'dereverb_mel_band_roformer_anvuew': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'dereverb_mel_band_roformer_anvuew.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt'),
            'download_urls': [
                'https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml',
                'https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt'
            ],
            'needs_conf_edit': True
        },
        'dereverb_mel_band_roformer_mono (by anvuew)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'dereverb_mel_band_roformer_anvuew.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt'),
            'download_urls': [
                'https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew.yaml',
                'https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt'
            ],
            'needs_conf_edit': True
        }
    },
    "Other Models": {
        'KARAOKE-MelBand-Roformer (by aufr33 & viperx)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_mel_band_roformer_karaoke.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt'),
            'download_urls': [
                'https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt',
                'https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/config_mel_band_roformer_karaoke.yaml'
            ],
            'needs_conf_edit': True
        },
        'OTHER-BS-Roformer_1053 (by viperx)': {
            'model_type': 'bs_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'model_bs_roformer_ep_937_sdr_10.5309.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model_bs_roformer_ep_937_sdr_10.5309.ckpt'),
            'download_urls': [
                'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_937_sdr_10.5309.ckpt',
                'https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_937_sdr_10.5309.yaml'
            ],
            'needs_conf_edit': True
        },
        'CROWD-REMOVAL-MelBand-Roformer (by aufr33)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'model_mel_band_roformer_crowd.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt'),
            'download_urls': [
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt',
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.4/model_mel_band_roformer_crowd.yaml'
            ],
            'needs_conf_edit': True
        },
        'CINEMATIC-BandIt_Plus (by kwatcharasupat)': {
            'model_type': 'bandit',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_dnr_bandit_bsrnn_multi_mus64.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model_bandit_plus_dnr_sdr_11.47.chpt'),
            'download_urls': [
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/config_dnr_bandit_bsrnn_multi_mus64.yaml',
                'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/model_bandit_plus_dnr_sdr_11.47.chpt'
            ],
            'needs_conf_edit': False
        },
        'DRUMSEP-MDX23C_DrumSep_6stem (by aufr33 & jarredou)': {
            'model_type': 'mdx23c',
            'config_path': os.path.join(CHECKPOINT_DIR, 'aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt'),
            'download_urls': [
                'https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt',
                'https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml'
            ],
            'needs_conf_edit': False
        },
        'bleed_suppressor_v1 (by unwa)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_bleed_suppressor_v1.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'bleed_suppressor_v1.ckpt'),
            'download_urls': [
                'https://huggingface.co/ASesYusuf1/MODELS/resolve/main/bleed_suppressor_v1.ckpt',
                'https://huggingface.co/ASesYusuf1/MODELS/resolve/main/config_bleed_suppressor_v1.yaml'
            ],
            'needs_conf_edit': True
        },
        'SYH99999/MelBandRoformerSYHFTB1_Model1 (by Amane)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model.ckpt'),
            'download_urls': [
                'https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/config.yaml',
                'https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/model.ckpt'
            ],
            'needs_conf_edit': True
        },
        'SYH99999/MelBandRoformerSYHFTB1_Model2 (by Amane)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model2.ckpt'),
            'download_urls': [
                'https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/config.yaml',
                'https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/model2.ckpt'
            ],
            'needs_conf_edit': True
        },
        'SYH99999/MelBandRoformerSYHFTB1_Model3 (by Amane)': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'model3.ckpt'),
            'download_urls': [
                'https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/config.yaml',
                'https://huggingface.co/SYH99999/MelBandRoformerSYHFTB1/resolve/main/model3.ckpt'
            ],
            'needs_conf_edit': True
        },
        'KaraokeGabox': {
            'model_type': 'mel_band_roformer',
            'config_path': os.path.join(CHECKPOINT_DIR, 'config_mel_band_roformer_karaoke.yaml'),
            'start_check_point': os.path.join(CHECKPOINT_DIR, 'KaraokeGabox.ckpt'),
            'download_urls': [
                'https://github.com/deton24/Colab-for-new-MDX_UVR_models/releases/download/v1.0.0/config_mel_band_roformer_karaoke.yaml',
                'https://huggingface.co/GaboxR67/MelBandRoformers/blob/main/melbandroformers/experimental/KaraokeGabox.ckpt'
            ],
            'needs_conf_edit': True
        }
    }
}

def get_model_config(clean_model=None, chunk_size=None, overlap=None):
    """Returns model type, config path, and checkpoint path for a given model name, downloading files if needed."""
    if clean_model is None:
        return {model_name for category in MODEL_CONFIGS.values() for model_name in category.keys()}
    
    for category in MODEL_CONFIGS.values():
        if clean_model in category:
            config = category[clean_model]
            for url in config['download_urls']:
                download_file(url)
            if config['needs_conf_edit'] and chunk_size is not None and overlap is not None:
                conf_edit(config['config_path'], chunk_size, overlap)
            return config['model_type'], config['config_path'], config['start_check_point']
    return "", "", ""

get_model_config.keys = lambda: {model_name for category in MODEL_CONFIGS.values() for model_name in category.keys()}
