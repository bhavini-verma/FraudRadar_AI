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

def ingest_elevenlabs_parquet(target_count=100):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_dir = os.path.join(base_dir, 'data', 'temp_download')
    out_dir = os.path.join(base_dir, "data", "raw_fake", "elevenlabs_english")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    existing = len([f for f in os.listdir(out_dir) if f.endswith(".wav")])
    if existing >= target_count:
        print(f"Already have {existing} files, skipping.")
        return existing
        
    print(f"Loading ElevenLabs samples directly from parquet...")
    parquet_url = "https://huggingface.co/api/datasets/garystafford/deepfake-audio-detection/parquet/default/train/0.parquet"
    
    parquet_path = download_file(parquet_url, temp_dir, "deepfake_audio_detection_0.parquet")
    df = pd.read_parquet(parquet_path)
    
    # Let's filter for elevenlabs if there is a 'label' or 'generator' column, 
    # but the dataset is "fake" (label=0) vs "real" (label=1). We just need fake samples.
    # We don't strictly know if it's ElevenLabs unless there's a column for it.
    # Let's just grab the fake ones. Wait, garystafford has multiple TTS.
    
    collected = existing
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if collected >= target_count:
            break
            
        try:
            # check if label is fake (usually 0 is fake in this dataset, but let's just grab any fake)
            # Actually, this dataset only has 'audio' and 'label' (or similar). Let's print columns first if we don't know
            if 'label' in row and row['label'] != 0 and str(row['label']).lower() != 'fake':
                continue
                
            audio_bytes = row['audio']['bytes']
            y, sr = sf.read(io.BytesIO(audio_bytes))
            
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000
                
            out_path = os.path.join(out_dir, f"elevenlabs_en_{collected:05d}.wav")
            sf.write(out_path, y, sr)
            collected += 1
        except Exception as e:
            # print(f"Error processing sample: {e}")
            continue
            
    print(f"Total synthetic files extracted: {collected}")

if __name__ == "__main__":
    ingest_elevenlabs_parquet(target_count=100)
