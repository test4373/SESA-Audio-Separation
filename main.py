import os
import threading
import urllib.request
import time
import sys
import random
import argparse
import librosa
from tqdm.auto import tqdm
import torch
import soundfile as sf
import torch.nn as nn
from datetime import datetime
import numpy as np
import shutil
from gui import create_interface

# pyngrok import (optional - only needed for ngrok sharing)
try:
    from pyngrok import ngrok
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False
    ngrok = None

from assets.i18n.i18n import I18nAuto  # I18nAuto'yu içe aktar

import warnings
warnings.filterwarnings("ignore")

def generate_random_port():
    """Generates a random port between 1000 and 9000."""
    return random.randint(1000, 9000)

def start_gradio(port, share=False):
    """Starts the Gradio interface with optional sharing."""
    demo = create_interface()
    demo.launch(
        server_port=port,
        server_name='0.0.0.0',
        share=share,
        allowed_paths=[os.path.join(os.path.expanduser("~"), "Music-Source-Separation", "input"), "/tmp", "/content"],
        inline=False
    )

def start_localtunnel(port, i18n):
    """Starts the Gradio interface with localtunnel sharing."""
    print(i18n("starting_localtunnel").format(port=port))
    os.system('npm install -g localtunnel &>/dev/null')
    
    with open('url.txt', 'w') as file:
        file.write('')
    os.system(f'lt --port {port} >> url.txt 2>&1 &')
    time.sleep(2)
    
    endpoint_ip = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n")
    with open('url.txt', 'r') as file:
        tunnel_url = file.read().replace("your url is: ", "").strip()

    print(i18n("share_link").format(url=tunnel_url))
    print(i18n("password_ip").format(ip=endpoint_ip))
    
    start_gradio(port, share=False)

def start_ngrok(port, ngrok_token, i18n):
    """Starts the Gradio interface with ngrok sharing."""
    if not NGROK_AVAILABLE:
        print("pyngrok modülü yüklü değil. 'pip install pyngrok' ile yükleyin.")
        sys.exit(1)
    print(i18n("starting_ngrok").format(port=port))
    try:
        ngrok.set_auth_token(ngrok_token)
        ngrok.kill()
        tunnel = ngrok.connect(port)
        print(i18n("ngrok_url").format(url=tunnel.public_url))
        
        start_gradio(port, share=False)
    except Exception as e:
        print(i18n("ngrok_error").format(error=str(e)))
        sys.exit(1)

def main(method="gradio", port=None, ngrok_token=""):
    """Main entry point for the application."""
    # I18nAuto'yu başlat
    i18n = I18nAuto()

    # Portu otomatik belirle veya kullanıcıdan geleni kullan
    port = port or generate_random_port()
    print(i18n("selected_port").format(port=port))

    # Paylaşım yöntemine göre işlem yap
    if method == "gradio":
        print(i18n("starting_gradio_with_sharing"))
        start_gradio(port, share=True)
    elif method == "localtunnel":
        start_localtunnel(port, i18n)
    elif method == "ngrok":
        if not ngrok_token:
            print(i18n("ngrok_token_required"))
            sys.exit(1)
        start_ngrok(port, ngrok_token, i18n)
    else:
        print(i18n("invalid_method"))
        sys.exit(1)

    # Sürekli çalışır durumda tut (gerekirse)
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print(i18n("process_stopped"))
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music Source Separation Web UI")
    parser.add_argument("--method", type=str, default="gradio", choices=["gradio", "localtunnel", "ngrok"], help="Sharing method (default: gradio)")
    parser.add_argument("--port", type=int, default=None, help="Server port (default: random between 1000-9000)")
    parser.add_argument("--ngrok-token", type=str, default="", help="Ngrok authentication token (required for ngrok)")
    args = parser.parse_args()
    
    main(method=args.method, port=args.port, ngrok_token=args.ngrok_token)
