import os
import glob
import librosa
import soundfile as sf
import numpy as np
import scipy.signal

def apply_codec(y, sr=16000):
    """Simulates telephony codec downsampling and upsampling."""
    y_8k = librosa.resample(y, orig_sr=sr, target_sr=8000)
    y_16k = librosa.resample(y_8k, orig_sr=8000, target_sr=sr)
    return y_16k

def apply_noise(y, snr_db):
    """Adds white noise at a specific Signal-to-Noise Ratio (SNR)."""
    sig_power = np.mean(y ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    return y + noise

def apply_replay(y, sr=16000):
    """Simulates a phone/laptop speaker replay with strict bandpass and saturation."""
    nyq = 0.5 * sr
    low = 300 / nyq
    high = 3400 / nyq
    b, a = scipy.signal.butter(5, [low, high], btype='band')
    y_filtered = scipy.signal.lfilter(b, a, y)
    
    # Slight saturation for cheap speaker distortion
    y_sat = np.tanh(y_filtered * 1.5) / 1.5
    return y_sat

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fake_dir = os.path.join(base_dir, 'data', 'raw_fake', 'fake')
    
    clean_files = sorted(glob.glob(os.path.join(fake_dir, 'elevenlabs_advanced_clean_*.wav')))
    if len(clean_files) != 100:
        print(f"Warning: Expected 100 clean files, found {len(clean_files)}")
    
    if len(clean_files) == 0:
        print("No clean files found! Please run generate_elevenlabs_advanced.py first.")
        return

    print(f"Augmenting {len(clean_files)} clean ElevenLabs WAVs into tiered degradation layers...")
    
    count_codec = 0
    count_noise = 0
    count_replay = 0
    
    for i, file_path in enumerate(clean_files):
        try:
            y, sr = librosa.load(file_path, sr=16000, mono=True)
            base_name = os.path.basename(file_path).replace('_clean_', '_')
            name_no_ext = os.path.splitext(base_name)[0]
            
            # 1. Codec Version
            y_codec = apply_codec(y, sr)
            codec_path = os.path.join(fake_dir, f"{name_no_ext}_codec.wav")
            sf.write(codec_path, y_codec, sr, subtype='PCM_16')
            count_codec += 1
            
            # 2. Noise Version
            if i < 50:
                snr = np.random.uniform(5, 20)
                y_noise = apply_noise(y, snr)
                noise_path = os.path.join(fake_dir, f"{name_no_ext}_noise.wav")
                sf.write(noise_path, y_noise, sr, subtype='PCM_16')
                count_noise += 1
                
            # 3. Replay Version
            if i >= 50:
                y_replay = apply_replay(y, sr)
                replay_path = os.path.join(fake_dir, f"{name_no_ext}_replay.wav")
                sf.write(replay_path, y_replay, sr, subtype='PCM_16')
                count_replay += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    print(f"Augmentation complete.")
    print(f"Generated Codec clips:  {count_codec}")
    print(f"Generated Noise clips:  {count_noise}")
    print(f"Generated Replay clips: {count_replay}")
    print(f"Total useful samples:   {len(clean_files) + count_codec + count_noise + count_replay}")

if __name__ == "__main__":
    main()
