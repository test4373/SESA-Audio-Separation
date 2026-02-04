import os
import re
import validators
import yt_dlp
import requests
import shutil
try:
    import gdown
except ImportError:
    os.system('pip install gdown')
    import gdown
try:
    from google.colab import drive
except ImportError:
    drive = None
from helpers import INPUT_DIR, COOKIE_PATH, clear_directory, clear_temp_folder, BASE_DIR
import yaml
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

def sanitize_filename(filename):
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', filename)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_')
    return sanitized

def download_callback(url, download_type='direct', cookie_file=None):
    # Clear temporary and input directories
    clear_temp_folder("/tmp", exclude_items=["gradio", "config.json"])
    clear_directory(INPUT_DIR)
    os.makedirs(INPUT_DIR, exist_ok=True)

    # Validate URL
    if not url or not isinstance(url, str) or not (url.startswith('http://') or url.startswith('https://')):
        return None, i18n("invalid_url"), None, None, None, None

    # Load cookie file
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
    drive_mounted = False

    # Mount Google Drive (optional)
    if drive is not None:
        try:
            # Check if already mounted first
            if os.path.exists('/content/drive/MyDrive'):
                drive_mounted = True
            else:
                drive.mount('/content/drive', force_remount=True)
                drive_mounted = True
        except AttributeError as ae:
            # Handle 'NoneType' object has no attribute 'kernel' error
            print(f"⚠️ Google Drive mount skipped (Colab kernel issue): {str(ae)}")
            print(i18n("continuing_without_google_drive"))
        except Exception as e:
            print(i18n("google_drive_mount_error").format(str(e)))
            print(i18n("continuing_without_google_drive"))

    # 1. Direct file links
    if any(url.endswith(ext) for ext in ['.wav', '.mp3', '.m4a', '.ogg', '.flac']):
        try:
            file_name = os.path.basename(url.split('?')[0])
            sanitized_base_name = sanitize_filename(os.path.splitext(file_name)[0])
            output_path = os.path.join(INPUT_DIR, f"{sanitized_base_name}{os.path.splitext(file_name)[1]}")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                if not file_name.endswith('.wav'):
                    wav_output = os.path.splitext(output_path)[0] + '.wav'
                    os.system(f'ffmpeg -i "{output_path}" -acodec pcm_s16le -ar 44100 "{wav_output}"')
                    if os.path.exists(wav_output):
                        os.remove(output_path)
                        output_path = wav_output
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    wav_path = output_path
                    download_success = True
                else:
                    raise Exception(i18n("file_size_zero_error"))
            else:
                raise Exception(i18n("direct_download_failed"))
        except Exception as e:
            error_msg = i18n("direct_download_error").format(str(e))
            print(error_msg)
            return None, error_msg, None, None, None, None

    # 2. Google Drive links
    elif 'drive.google.com' in url:
        try:
            file_id = re.search(r'/d/([^/]+)', url) or re.search(r'id=([^&]+)', url)
            if not file_id:
                raise ValueError(i18n("invalid_google_drive_url"))
            file_id = file_id.group(1)
            output_path = os.path.join(INPUT_DIR, "drive_download.wav")
            gdown.download(f'https://drive.google.com/uc?id={file_id}', output_path, quiet=True)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                sanitized_base_name = sanitize_filename("drive_download")
                sanitized_output_path = os.path.join(INPUT_DIR, f"{sanitized_base_name}.wav")
                os.rename(output_path, sanitized_output_path)
                wav_path = sanitized_output_path
                download_success = True
            else:
                raise Exception(i18n("file_size_zero_error"))
        except Exception as e:
            error_msg = i18n("google_drive_error").format(str(e))
            print(error_msg)
            return None, error_msg, None, None, None, None

    # 3. YouTube and other media links
    else:
        # First try: iOS/Android without cookies (best for bot protection bypass)
        ydl_opts_nocookie = {
            'format': 'ba[ext=m4a]/ba[ext=webm]/ba/b',
            'outtmpl': os.path.join(INPUT_DIR, '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '0'
            }],
            'nocheckcertificate': True,
            'ignoreerrors': False,
            'retries': 3,
            'extractor_retries': 3,
            'extractor_args': {
                'youtube': {
                    'player_client': ['ios', 'android'],
                    'player_skip': ['webpage', 'configs']
                }
            },
            'http_headers': {
                'User-Agent': 'com.google.ios.youtube/19.09.3 (iPhone14,3; U; CPU iOS 15_6 like Mac OS X)',
                'Accept-Language': 'en-US,en;q=0.9'
            }
        }
        
        # Second try: web client with cookies if available
        ydl_opts_cookie = {
            'format': 'ba[ext=m4a]/ba[ext=webm]/ba/b',
            'outtmpl': os.path.join(INPUT_DIR, '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '0'
            }],
            'cookiefile': COOKIE_PATH,
            'nocheckcertificate': True,
            'ignoreerrors': False,
            'retries': 3,
            'extractor_retries': 3,
            'extractor_args': {
                'youtube': {
                    'player_client': ['web', 'tv_embedded'],
                    'player_skip': ['configs']
                }
            }
        }
        
        # Try without cookies first
        info_dict = None
        temp_path = None
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts_nocookie) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                if info_dict:
                    temp_path = ydl.prepare_filename(info_dict)
        except Exception as e:
            # If no cookies available or first method failed, try with cookies
            if os.path.exists(COOKIE_PATH):
                print(f"⚠️ First attempt failed, trying with cookies...")
                try:
                    with yt_dlp.YoutubeDL(ydl_opts_cookie) as ydl:
                        info_dict = ydl.extract_info(url, download=True)
                        if info_dict:
                            temp_path = ydl.prepare_filename(info_dict)
                except Exception as e2:
                    raise e2
            else:
                raise e
        
        try:
            # Check if extraction was successful
            if info_dict is None:
                raise Exception(i18n("youtube_extraction_failed") if "youtube_extraction_failed" in dir(i18n) else "YouTube extraction failed. Please try updating yt-dlp: pip install -U yt-dlp")
            
            base_name = os.path.splitext(os.path.basename(temp_path))[0]
            sanitized_base_name = sanitize_filename(base_name)
            wav_path = os.path.join(INPUT_DIR, f"{sanitized_base_name}.wav")
            temp_wav = os.path.splitext(temp_path)[0] + '.wav'
            if os.path.exists(temp_wav):
                os.rename(temp_wav, wav_path)
                download_success = True
            else:
                raise Exception(i18n("wav_conversion_failed"))
        except Exception as e:
            error_msg = i18n("download_error").format(str(e))
            # Add hint for yt-dlp update if it's a YouTube issue
            if 'youtube' in url.lower() or 'youtu.be' in url.lower():
                error_msg += "\n\n💡 Try: pip install -U yt-dlp"
            print(error_msg)
            return None, error_msg, None, None, None, None

    # Post-download processing
    if download_success and wav_path:
        for f in os.listdir(INPUT_DIR):
            if f != os.path.basename(wav_path):
                os.remove(os.path.join(INPUT_DIR, f))
        
        if drive_mounted:
            try:
                drive_path = os.path.join('/content/drive/My Drive', os.path.basename(wav_path))
                shutil.copy(wav_path, drive_path)
                print(i18n("file_copied_to_drive").format(drive_path))
            except Exception as e:
                print(i18n("copy_to_drive_error").format(str(e)))
        else:
            print(i18n("skipping_drive_copy_no_mount"))

        return (
            wav_path,
            i18n("download_success"),
            wav_path,
            wav_path,
            wav_path,
            wav_path
        )
    
    return None, i18n("download_failed"), None, None, None, None
