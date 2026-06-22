"""
fraudradar_ai — Hindi Fake Generator Diversification
==================================================
Downloads additional Hindi fake audio from multiple synthesis architectures
to prevent the model from overfitting to FreeVC artifacts.

Generators added:
  1. xtts_v2     — from IndicSynth parquet shard (direct HF download)
  2. edge-tts    — Microsoft Azure Neural TTS (free, no API key)
  3. gTTS        — Google Translate TTS (free, parametric)
"""

import os
import io
import sys
import asyncio

import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE_DIR = os.path.join(BASE_DIR, "data", "raw_fake")
TEMP_DIR = os.path.join(BASE_DIR, "data", "temp_download")

# Hindi sentences for TTS generation (diverse phonemes, lengths, topics)
HINDI_SENTENCES = [
    "भारत एक महान देश है और यहाँ विविधता में एकता है।",
    "आज मौसम बहुत अच्छा है, बाहर चलते हैं।",
    "कृपया मुझे एक गिलास पानी दे दीजिए।",
    "मैं कल सुबह दिल्ली जा रहा हूँ, ट्रेन पकड़नी है।",
    "हमारे देश की संस्कृति बहुत समृद्ध और प्राचीन है।",
    "बच्चों को अच्छी शिक्षा देना हर माता-पिता का कर्तव्य है।",
    "प्रौद्योगिकी ने हमारे जीवन को पूरी तरह बदल दिया है।",
    "क्या आप मुझे बता सकते हैं कि निकटतम अस्पताल कहाँ है?",
    "इस साल मानसून समय पर आया और किसानों को बहुत राहत मिली।",
    "संगीत एक ऐसी कला है जो आत्मा को सुकून देती है।",
    "हमें पर्यावरण की रक्षा के लिए पेड़ लगाने चाहिए।",
    "विज्ञान और तकनीक के बिना आधुनिक जीवन संभव नहीं है।",
    "गांधी जी ने अहिंसा का मार्ग अपनाकर देश को आजादी दिलाई।",
    "रात को जल्दी सोना और सुबह जल्दी उठना स्वास्थ्य के लिए अच्छा है।",
    "कंप्यूटर विज्ञान आज के युग का सबसे महत्वपूर्ण विषय बन गया है।",
    "हिमालय की चोटियों पर बर्फ पिघलने लगी है जो चिंता का विषय है।",
    "मेरी दादी बहुत अच्छी कहानियाँ सुनाती थीं बचपन में।",
    "भारतीय रेलवे दुनिया का सबसे बड़ा रेल नेटवर्क में से एक है।",
    "योग और ध्यान से मानसिक शांति और शारीरिक स्वास्थ्य मिलता है।",
    "आर्टिफिशियल इंटेलिजेंस भविष्य की सबसे बड़ी क्रांति होगी।",
    "मुंबई की बारिश में सड़कों पर पानी भर जाता है।",
    "राजस्थान के रेगिस्तान में ऊँटों की सवारी बहुत मजेदार होती है।",
    "स्वस्थ खाना खाओ, नियमित व्यायाम करो, और खुश रहो।",
    "टेलीफोन पर बात करते समय सावधान रहना चाहिए।",
    "डिजिटल भुगतान ने नकदी की जरूरत को काफी कम कर दिया है।",
    "भारतीय क्रिकेट टीम ने विश्व कप में शानदार प्रदर्शन किया।",
    "गर्मियों में आम का मौसम आता है और बच्चे बहुत खुश होते हैं।",
    "पुस्तकें हमारी सबसे अच्छी मित्र हैं, इन्हें पढ़ते रहना चाहिए।",
    "सरकार ने नई शिक्षा नीति की घोषणा की है।",
    "ताजमहल दुनिया के सात अजूबों में से एक है।",
    "अंतरिक्ष विज्ञान में भारत ने बहुत तरक्की की है।",
    "किसान हमारे अन्नदाता हैं, हमें उनका सम्मान करना चाहिए।",
    "नदियों को प्रदूषण से बचाना हम सबकी जिम्मेदारी है।",
    "आज रात चाँद बहुत सुंदर दिख रहा है आसमान में।",
    "बैंक में खाता खोलने के लिए आधार कार्ड जरूरी है।",
    "सर्दियों में गरम चाय पीना बहुत अच्छा लगता है।",
    "इंटरनेट ने दुनिया को एक गाँव बना दिया है।",
    "हर नागरिक का कर्तव्य है कि वह कानून का पालन करे।",
    "भारत में अनेक भाषाएँ बोली जाती हैं और सब की अपनी खूबसूरती है।",
    "खेलकूद से शरीर स्वस्थ रहता है और मन प्रसन्न रहता है।",
    "वायु प्रदूषण एक गंभीर समस्या है जिससे निपटना जरूरी है।",
    "त्योहार हमारे जीवन में खुशियाँ और उत्साह लाते हैं।",
    "सोशल मीडिया का उपयोग सोच-समझकर करना चाहिए।",
    "भारतीय खाना पूरी दुनिया में अपने स्वाद के लिए मशहूर है।",
    "पानी बचाओ, जीवन बचाओ, यह हम सबकी जिम्मेदारी है।",
    "स्कूल में अध्यापकों का बहुत महत्वपूर्ण योगदान होता है।",
    "रेडियो सुनना मुझे बहुत पसंद है, खासकर पुराने गाने।",
    "मोबाइल फोन आज हर किसी की जरूरत बन गया है।",
    "वैश्विक तापमान बढ़ने से ग्लेशियर तेजी से पिघल रहे हैं।",
    "हमें अपनी मातृभाषा का सम्मान और संरक्षण करना चाहिए।",
]


# ─── 1. Ingest XTTS_V2 from IndicSynth (download parquet shard directly) ────

def ingest_xtts_from_indicsynth(target_count=100):
    """Download xtts_v2 samples from IndicSynth Hindi by loading a later parquet shard."""
    out_dir = os.path.join(FAKE_DIR, "indicsynth_hindi_xtts")
    os.makedirs(out_dir, exist_ok=True)

    existing = len([f for f in os.listdir(out_dir) if f.endswith(".wav")])
    if existing >= target_count:
        print(f"[xtts_v2] Already have {existing} files, skipping.")
        return existing

    print(f"[xtts_v2] Loading IndicSynth Hindi parquet shard 55 (direct download, no streaming)...")

    # Read parquet shard 55 containing 100% xtts_v2 files
    shard_url = "https://huggingface.co/api/datasets/vdivyasharma/IndicSynth/parquet/Hindi/train/55.parquet"
    df = pd.read_parquet(shard_url)

    collected = existing
    print(f"[xtts_v2] Extracting {target_count - collected} samples from shard 55...")
    
    for idx, row in df.iterrows():
        if collected >= target_count:
            break

        out_path = os.path.join(out_dir, f"hindi_xtts_{collected:05d}.wav")
        if os.path.exists(out_path):
            collected += 1
            continue

        try:
            audio_data = row["audio"]
            audio_bytes = audio_data["bytes"]

            y, sr = sf.read(io.BytesIO(audio_bytes))

            # Resample to 16kHz
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000

            sf.write(out_path, y, sr)
            collected += 1

            if collected % 10 == 0:
                print(f"  [xtts_v2] Saved {collected}/{target_count}")

        except Exception as e:
            print(f"  [xtts_v2] Failed to process sample at index {idx}: {e}")
            continue

    print(f"[xtts_v2] Total saved: {collected}")
    return collected


# ─── 2. Generate Hindi fakes using Edge TTS (Microsoft Neural TTS) ──────────

async def generate_edge_tts_hindi(target_count=100):
    """Generate Hindi TTS using Microsoft Edge TTS (free, neural voices)."""
    import edge_tts

    out_dir = os.path.join(FAKE_DIR, "edge_tts_hindi")
    os.makedirs(out_dir, exist_ok=True)

    existing = len([f for f in os.listdir(out_dir) if f.endswith(".wav")])
    if existing >= target_count:
        print(f"[edge-tts] Already have {existing} files, skipping.")
        return existing

    # Hindi neural voices available in Edge TTS
    hindi_voices = [
        "hi-IN-SwaraNeural",    # Female
        "hi-IN-MadhurNeural",   # Male
    ]

    collected = existing
    sentence_idx = existing

    for i in range(existing, target_count):
        sentence = HINDI_SENTENCES[sentence_idx % len(HINDI_SENTENCES)]
        voice = hindi_voices[i % len(hindi_voices)]
        sentence_idx += 1

        out_path = os.path.join(out_dir, f"hindi_edge_{i:05d}.wav")
        if os.path.exists(out_path):
            collected += 1
            continue

        # Edge TTS outputs MP3 by default
        temp_mp3 = out_path.replace(".wav", ".mp3")

        try:
            communicate = edge_tts.Communicate(sentence, voice)
            await communicate.save(temp_mp3)

            # Convert MP3 to 16kHz mono WAV
            y, sr = librosa.load(temp_mp3, sr=16000, mono=True)
            sf.write(out_path, y, 16000)

            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)

            collected += 1

            if collected % 10 == 0:
                print(f"  [edge-tts] Generated {collected}/{target_count}")

        except Exception as e:
            print(f"  [edge-tts] Failed on sentence {i}: {e}")
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
            continue

    print(f"[edge-tts] Total generated: {collected}")
    return collected


# ─── 3. Generate Hindi fakes using Google TTS (parametric) ──────────────────

def generate_gtts_hindi(target_count=100):
    """Generate Hindi TTS using Google Translate TTS (free, parametric)."""
    from gtts import gTTS

    out_dir = os.path.join(FAKE_DIR, "gtts_hindi")
    os.makedirs(out_dir, exist_ok=True)

    existing = len([f for f in os.listdir(out_dir) if f.endswith(".wav")])
    if existing >= target_count:
        print(f"[gTTS] Already have {existing} files, skipping.")
        return existing

    collected = existing
    sentence_idx = existing

    for i in range(existing, target_count):
        sentence = HINDI_SENTENCES[sentence_idx % len(HINDI_SENTENCES)]
        sentence_idx += 1

        out_path = os.path.join(out_dir, f"hindi_gtts_{i:05d}.wav")
        if os.path.exists(out_path):
            collected += 1
            continue

        temp_mp3 = out_path.replace(".wav", ".mp3")

        try:
            tts = gTTS(text=sentence, lang="hi")
            tts.save(temp_mp3)

            # Convert to 16kHz mono WAV
            y, sr = librosa.load(temp_mp3, sr=16000, mono=True)
            sf.write(out_path, y, 16000)

            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)

            collected += 1

            if collected % 10 == 0:
                print(f"  [gTTS] Generated {collected}/{target_count}")

        except Exception as e:
            print(f"  [gTTS] Failed on sentence {i}: {e}")
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
            continue

    print(f"[gTTS] Total generated: {collected}")
    return collected


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("fraudradar_ai — Hindi Fake Generator Diversification")
    print("=" * 70)
    sys.stdout.flush()

    results = {}

    # Phase 1: Edge TTS first (fastest, no download needed)
    print("\n--- Phase 1: Edge TTS (Microsoft Neural) ---")
    sys.stdout.flush()
    results["edge_tts"] = asyncio.run(generate_edge_tts_hindi(target_count=100))

    # Phase 2: gTTS (Google Parametric, also fast)
    print("\n--- Phase 2: gTTS (Google Parametric) ---")
    sys.stdout.flush()
    results["gtts"] = generate_gtts_hindi(target_count=100)

    # Phase 3: XTTS_V2 from IndicSynth (streaming, slower)
    print("\n--- Phase 3: XTTS_V2 from IndicSynth ---")
    sys.stdout.flush()
    results["xtts_v2"] = ingest_xtts_from_indicsynth(target_count=100)

    # Summary
    print("\n" + "=" * 70)
    print("=== DIVERSIFICATION COMPLETE ===")
    print("=" * 70)

    total_new = 0
    for gen, count in results.items():
        print(f"  {gen:>15s}: {count} files")
        total_new += count

    print(f"\n  Total new Hindi fakes: {total_new}")
    print(f"  Existing FreeVC Hindi: 657 files")
    print(f"  Grand total Hindi fake: {total_new + 657}")
    print()
    print("New directories created:")
    for name in ["indicsynth_hindi_xtts", "edge_tts_hindi", "gtts_hindi"]:
        path = os.path.join(FAKE_DIR, name)
        if os.path.exists(path):
            n = len([f for f in os.listdir(path) if f.endswith(".wav")])
            print(f"  {path} ({n} files)")


if __name__ == "__main__":
    main()
