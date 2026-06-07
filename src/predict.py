"""
VoiceGuard — Inference
src/predict.py

Usage:
    python src/predict.py --audio path/to/file.wav
    python src/predict.py --audio path/to/file.wav --model models/model.pkl
"""

import argparse
import numpy as np
import librosa
import joblib
import os

# ── CONFIG ──────────────────────────────────────────
SR       = 16000
N_MFCC   = 40
DURATION = 3
# ────────────────────────────────────────────────────

# Risk thresholds (probability of being FAKE)
LOW_MAX  = 0.40   # 0–40%  → LOW
MED_MAX  = 0.70   # 40–70% → MEDIUM
# above 70%       → HIGH


def extract_features(file_path):
    """Same pipeline as extract.py — must stay in sync."""
    y, sr = librosa.load(file_path, sr=SR, duration=DURATION, mono=True)

    mfcc         = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean    = np.mean(mfcc, axis=1)
    mfcc_std     = np.std(mfcc, axis=1)

    mel          = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db       = librosa.power_to_db(mel, ref=np.max)
    mel_mean     = np.mean(mel_db)
    mel_std      = np.std(mel_db)

    zcr          = librosa.feature.zero_crossing_rate(y)
    zcr_mean     = np.mean(zcr)
    zcr_std      = np.std(zcr)

    contrast     = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std  = np.std(contrast, axis=1)

    chroma       = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean  = np.mean(chroma, axis=1)
    chroma_std   = np.std(chroma, axis=1)

    f0, _, _     = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    f0_valid     = f0[~np.isnan(f0)]
    pitch_std    = np.std(f0_valid) if len(f0_valid) > 0 else 0.0

    rms          = librosa.feature.rms(y=y)
    rms_mean     = np.mean(rms)
    rms_std      = np.std(rms)

    harmonic     = librosa.effects.harmonic(y)
    harmonic_ratio = np.mean(np.abs(harmonic)) / (np.mean(np.abs(y)) + 1e-6)

    features = np.concatenate([
        mfcc_mean, mfcc_std,
        [mel_mean], [mel_std],
        [zcr_mean], [zcr_std],
        contrast_mean, contrast_std,
        chroma_mean, chroma_std,
        [pitch_std],
        [rms_mean], [rms_std],
        [harmonic_ratio],
    ])
    features = np.pad(features, (0, 128 - len(features)))
    return features


def risk_label(prob_fake):
    if prob_fake < LOW_MAX:
        return "🟢 LOW RISK", prob_fake
    elif prob_fake < MED_MAX:
        return "🟡 MEDIUM RISK", prob_fake
    else:
        return "🔴 HIGH RISK", prob_fake


def predict(audio_path, model_path="models/model.pkl", scaler_path="models/scaler.pkl"):
    # Load model and scaler
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path} — run train.py first")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path} — run train.py first")

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Extract + scale features
    print(f"Analyzing: {audio_path}")
    feat = extract_features(audio_path).reshape(1, -1)
    feat = scaler.transform(feat)

    # Predict
    prob_fake = model.predict_proba(feat)[0][1]
    label, confidence = risk_label(prob_fake)

    print(f"\nRESULT: {label}")
    print(f"  Confidence (fake probability): {confidence*100:.1f}%")

    if label.startswith("🔴"):
        print("  ACTION: Block call + raise alert")
    elif label.startswith("🟡"):
        print("  ACTION: Trigger secondary verification")
    else:
        print("  ACTION: Call proceeds normally")

    return label, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoiceGuard — predict real vs AI voice")
    parser.add_argument("--audio",  required=True,             help="Path to WAV file")
    parser.add_argument("--model",  default="models/model.pkl",  help="Path to model.pkl")
    parser.add_argument("--scaler", default="models/scaler.pkl", help="Path to scaler.pkl")
    args = parser.parse_args()

    predict(args.audio, args.model, args.scaler)        