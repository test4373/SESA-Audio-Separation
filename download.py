import os
import re
import validators
import yt_dlp
import torch
from urllib.parse import quote
from helpers import INPUT_DIR, COOKIE_PATH, clear_directory, clear_temp_folder, BASE_DIR
import yaml
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

def download_callback(url, download_type='direct', cookie_file=None):
    clear_temp_folder("/tmp", exclude_items=["gradio", "config.json"])
    clear_directory(INPUT_DIR)
    os.makedirs(INPUT_DIR, exist_ok=True)

    if not validators.url(url):
        return None, i18n("invalid_url"), None, None, None, None

    if cookie_file is not None:
        try:
            with open(cookie_file.name, "rb") as f:
                cookie_content = f.read()
            with open(COOKIE_PATH, "wb") as f:
                f.write(cookie_content)
            print(i18n("cookie_file_updated"))
        except Exception as e:
            print(i18n("cookie_installation_error").format(str(e)))

    wav_path = None
    download_success = False

    if download_type == 'drive':
        try:
            file_id = re.search(r'/d/([^/]+)', url).group(1) if '/d/' in url else url.split('id=')[-1]
            output_path = os.path.join(INPUT_DIR, "drive_download.wav")
            gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=True)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                wav_path = output_path
                download_success = True
            else:
                raise Exception(i18n("file_size_zero_error"))
        except Exception as e:
            error_msg = i18n("google_drive_error").format(str(e))
            print(error_msg)
            return None, error_msg, None, None, None, None
    else:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(INPUT_DIR, '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '0'
            }],
            'cookiefile': COOKIE_PATH if os.path.exists(COOKIE_PATH) else None,
            'nocheckcertificate': True,
            'ignoreerrors': True,
            'retries': 3
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                temp_path = ydl.prepare_filename(info_dict)
                wav_path = os.path.splitext(temp_path)[0] + '.wav'
                if os.path.exists(wav_path):
                    download_success = True
                else:
                    raise Exception(i18n("wav_conversion_failed"))
        except Exception as e:
            error_msg = i18n("download_error").format(str(e))
            print(error_msg)
            return None, error_msg, None, None, None, None

    if download_success and wav_path:
        for f in os.listdir(INPUT_DIR):
            if f != os.path.basename(wav_path):
                os.remove(os.path.join(INPUT_DIR, f))
        return (
            wav_path,
            i18n("download_success"),
            wav_path,
            wav_path,
            wav_path,
            wav_path
        )
    return None, i18n("download_failed"), None, None, None, None
