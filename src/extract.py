"""
VoiceGuard — Feature Extraction Pipeline
src/extract.py

Extracts 128-dim feature vector from each audio clip.
Output: features/features.csv
"""

import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

# ── CONFIG ──────────────────────────────────────────
REAL_DIR  = "data/real"
FAKE_DIR  = "data/fake"
OUT_CSV   = "features/features.csv"
SR        = 16000   # sample rate
N_MFCC    = 40      # MFCC coefficients
DURATION  = 3       # seconds to load per clip (keeps it fast)
# ────────────────────────────────────────────────────


def extract_features(file_path):
    """
    Load audio and return a 1D feature vector (128 dims).
    Returns None if file is unreadable.
    """
    try:
        y, sr = librosa.load(file_path, sr=SR, duration=DURATION, mono=True)

        # 1. MFCC — 40 coefficients → mean + std = 80 features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std  = np.std(mfcc, axis=1)

        # 2. Mel Spectrogram — mean + std = 2 features
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_mean = np.mean(mel_db)
        mel_std  = np.std(mel_db)

        # 3. Zero Crossing Rate — mean + std = 2 features
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std  = np.std(zcr)

        # 4. Spectral Contrast — 7 bands → mean + std = 14 features
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std  = np.std(contrast, axis=1)

        # 5. Chroma Features — 12 → mean + std = 24 features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std  = np.std(chroma, axis=1)

        # 6. Pitch (F0) jitter proxy — std of F0 = 1 feature
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        f0_valid = f0[~np.isnan(f0)]
        pitch_std = np.std(f0_valid) if len(f0_valid) > 0 else 0.0

        # 7. RMS Energy (noise floor proxy) — mean + std = 2 features
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std  = np.std(rms)

        # 8. Harmonic ratio — 1 feature
        harmonic = librosa.effects.harmonic(y)
        harmonic_ratio = np.mean(np.abs(harmonic)) / (np.mean(np.abs(y)) + 1e-6)

        # ── Assemble vector ──────────────────────────
        features = np.concatenate([
            mfcc_mean,          # 40
            mfcc_std,           # 40
            [mel_mean],         # 1
            [mel_std],          # 1
            [zcr_mean],         # 1
            [zcr_std],          # 1
            contrast_mean,      # 7
            contrast_std,       # 7
            chroma_mean,        # 12
            chroma_std,         # 12
            [pitch_std],        # 1
            [rms_mean],         # 1
            [rms_std],          # 1
            [harmonic_ratio],   # 1
        ])
        # Total: 40+40+1+1+1+1+7+7+12+12+1+1+1+1 = 126 dims
        # Pad to 128 with zeros for future use
        features = np.pad(features, (0, 128 - len(features)))
        return features

    except Exception as e:
        print(f"  [SKIP] {file_path} — {e}")
        return None


def process_folder(folder, label):
    """Process all WAV files in a folder, return list of feature rows."""
    rows = []
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    print(f"\nProcessing {len(files)} files from {folder} (label={label})")
    for fname in tqdm(files):
        path = os.path.join(folder, fname)
        feat = extract_features(path)
        if feat is not None:
            rows.append(list(feat) + [label])
    return rows


def main():
    os.makedirs("features", exist_ok=True)

    rows = []
    rows += process_folder(REAL_DIR, label=0)   # 0 = real
    rows += process_folder(FAKE_DIR, label=1)   # 1 = fake

    # Column names
    cols = [f"f{i}" for i in range(128)] + ["label"]
    df = pd.DataFrame(rows, columns=cols)

    df.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Done — {len(df)} clips processed")
    print(f"   Real: {len(df[df.label==0])}  |  Fake: {len(df[df.label==1])}")
    print(f"   Saved → {OUT_CSV}")


if __name__ == "__main__":
    main()