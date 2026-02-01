# Kaggle için gerekli kütüphaneleri yükleme
!pip install yt-dlp validators requests
import os
import re
import validators
import yt_dlp
import requests
import shutil

# Kaggle dizin ayarları
BASE_DIR = "/kaggle/working/"
INPUT_DIR = os.path.join(BASE_DIR, "input")
COOKIE_PATH = os.path.join(BASE_DIR, "cookies.txt")

# Giriş dizinini oluştur
os.makedirs(INPUT_DIR, exist_ok=True)

# Basit bir i18n yerine geçici hata mesajları
def i18n(key):
    messages = {
        "invalid_url": "Geçersiz URL.",
        "cookie_file_updated": "Çerez dosyası güncellendi.",
        "cookie_installation_error": "Çerez dosyası yükleme hatası: {}",
        "file_size_zero_error": "Dosya boyutu sıfır.",
        "direct_download_failed": "Doğrudan indirme başarısız.",
        "direct_download_error": "Doğrudan indirme hatası: {}",
        "google_drive_error": "Google Drive indirme hatası: {}",
        "download_error": "İndirme hatası: {}",
        "wav_conversion_failed": "WAV dönüşümü başarısız.",
        "download_success": "İndirme başarılı.",
        "download_failed": "İndirme başarısız."
    }
    return messages.get(key, key)

# helpers.py'deki fonksiyonların sadeleştirilmiş hali
def clear_directory(directory):
    if os.path.exists(directory):
        for f in os.listdir(directory):
            file_path = os.path.join(directory, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

def clear_temp_folder(directory, exclude_items=[]):
    if os.path.exists(directory):
        for f in os.listdir(directory):
            if f not in exclude_items:
                file_path = os.path.join(directory, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)

def download_callback(url, download_type='direct', cookie_file=None):
    # Geçici ve giriş dizinlerini temizle
    clear_temp_folder("/tmp", exclude_items=["gradio", "config.json"])
    clear_directory(INPUT_DIR)
    os.makedirs(INPUT_DIR, exist_ok=True)

    # URL'nin temel doğruluğunu kontrol et
    if not url or not isinstance(url, str) or not (url.startswith('http://') or url.startswith('https://')):
        return None, i18n("invalid_url"), None, None, None, None

    # Çerez dosyasını yükle
    if cookie_file is not None:
        try:
            with open(cookie_file, "rb") as f:
                cookie_content = f.read()
            with open(COOKIE_PATH, "wb") as f:
                f.write(cookie_content)
            print(i18n("cookie_file_updated"))
        except Exception as e:
            print(i18n("cookie_installation_error").format(str(e)))

    wav_path = None
    download_success = False

    # 1. Discord CDN veya doğrudan dosya linki kontrolü
    if any(url.endswith(ext) for ext in ['.wav', '.mp3', '.m4a', '.ogg', '.flac']):
        try:
            file_name = os.path.basename(url.split('?')[0])
            output_path = os.path.join(INPUT_DIR, file_name)
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                # Dosya WAV değilse, FFmpeg ile WAV'a dönüştür
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

    # 2. Google Drive linki yerine Kaggle veri seti kullanımı
    # Not: Google Drive yerine Kaggle veri setinden dosya yüklenmesi önerilir
    # Örnek: /kaggle/input/your-dataset/sample_audio.wav
    # Bu kısmı atlayacağız, çünkü Kaggle'da Google Drive bağlantısı desteklenmez

    # 3. YouTube ve diğer medya linkleri için yt-dlp
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

    # Başarılı indirme sonrası işlemleri
    if download_success and wav_path:
        # Gereksiz dosyaları temizle
        for f in os.listdir(INPUT_DIR):
            if f != os.path.basename(wav_path):
                os.remove(os.path.join(INPUT_DIR, f))
        
        # Kaggle'da Google Drive yerine /kaggle/working/ dizinine kaydet
        print(f"Dosya kaydedildi: {wav_path}")

        return (
            wav_path,
            i18n("download_success"),
            wav_path,
            wav_path,
            wav_path,
            wav_path
        )
    
    return None, i18n("download_failed"), None, None, None, None

# Örnek kullanım
# url = "https://example.com/sample.mp3"
# wav_path, message, _, _, _, _ = download_callback(url)
# print(message, wav_path)
