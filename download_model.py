import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise error for bad status codes
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

# Create models directory if not exists
os.makedirs('models', exist_ok=True)

# Wav2Lip GAN model URL
model_url = "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/ymaheshwari_iiit_ac_in/EQPbABVqK9FJqAkNqz3wxZABLa9pAiKNpdsi-p_vH_1w2A?download=1"
download_file(model_url, "models/wav2lip_gan.pth")
