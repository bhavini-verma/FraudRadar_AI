import os
import io
import subprocess
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm

def download_file(url, dest_dir, dest_filename):
    aria2c_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               'aria2', 'aria2-1.36.0-win-64bit-build1', 'aria2c.exe')
    dest_path = os.path.join(dest_dir, dest_filename)
    
    if os.path.exists(dest_path):
        print(f"{dest_filename} already exists, skipping download.")
        return dest_path
        
    print(f"Downloading {url} to {dest_path} using aria2c...")
    cmd = [
        aria2c_path,
        "-c", "-x", "16", "-s", "16",
        "--max-connection-per-server=16",
        "--retry-wait=3", "--max-tries=10",
        "-d", dest_dir, "-o", dest_filename,
        url
    ]
    subprocess.run(cmd, check=True)
    return dest_path

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_dir = os.path.join(base_dir, 'data', 'temp_download')
    real_dir = os.path.join(base_dir, 'data', 'raw_real', 'hindi_common_voice')
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)
    
    # Target counts
    target_count = 957
    existing = len([f for f in os.listdir(real_dir) if f.startswith("hindi_real_") and f.endswith(".wav")])
    
    if existing >= target_count:
        print(f"Already have {existing} real Hindi clips. No need to balance.")
        return
        
    to_extract = target_count - existing
    print(f"Need {to_extract} more clips to reach {target_count}.")
    
    train_url = "https://huggingface.co/api/datasets/google/fleurs/parquet/hi_in/train/0.parquet"
    print("Downloading FLEURS Hindi train split...")
    train_parquet = download_file(train_url, temp_dir, "fleurs_hindi_train.parquet")
    
    print("Extracting audio...")
    df = pd.read_parquet(train_parquet)
    
    total_extracted = existing
    
    for idx, row in tqdm(df.iterrows(), total=min(len(df), to_extract)):
        if total_extracted >= target_count:
            break
            
        try:
            audio_bytes = row['audio']['bytes']
            y, sr = sf.read(io.BytesIO(audio_bytes))
            
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000
                
            out_path = os.path.join(real_dir, f"hindi_real_{total_extracted:05d}.wav")
            sf.write(out_path, y, sr)
            total_extracted += 1
        except Exception as e:
            print(f"Failed to process row {idx}: {e}")
            continue
            
    print(f"Successfully balanced dataset! Total real clips: {total_extracted}")

if __name__ == "__main__":
    main()
