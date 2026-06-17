import os
import glob
import librosa
import soundfile as sf

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fake_dir = os.path.join(base_dir, 'data', 'raw_fake', 'fake')
    
    clean_files = sorted(glob.glob(os.path.join(fake_dir, 'elevenlabs_advanced_clean_*.wav')))
    print(f"Fixing format for {len(clean_files)} files to standard 16-bit PCM...")
    
    count = 0
    for file_path in clean_files:
        try:
            y, sr = librosa.load(file_path, sr=None, mono=True)
            sf.write(file_path, y, sr, subtype='PCM_16')
            count += 1
        except Exception as e:
            print(f"Failed {file_path}: {e}")
            
    print(f"Successfully fixed {count} files. They should now be playable anywhere.")

if __name__ == "__main__":
    main()
