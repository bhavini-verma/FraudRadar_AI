import os
import requests
import json
import time

API_KEY = "sk_f50430554e8f8a0b68ad6fd5ba46e4e03e69c0c82d07283e"
HEADERS = {
    "xi-api-key": API_KEY,
    "Content-Type": "application/json"
}

HINDI_SENTENCES = [
    "आपका नाम क्या है?",
    "मैं आज बहुत खुश हूँ।",
    "क्या आप मुझे एक गिलास पानी दे सकते हैं?",
    "यह किताब किसने लिखी है?",
    "मुझे हिंदी गाने सुनना बहुत पसंद है।",
    "कल हम फिल्म देखने जाएंगे।",
    "भारत की राजधानी दिल्ली है।",
    "आपको कौन सा रंग सबसे ज्यादा पसंद है?",
    "कृपया धीरे बोलिए, मुझे समझ नहीं आ रहा है।",
    "आज का मौसम बहुत सुहावना है।",
    "मैंने अपना काम खत्म कर लिया है।",
    "तुम कल कहाँ गए थे?",
    "क्या आप मेरी मदद कर सकते हैं?",
    "मुझे भूख लगी है, कुछ खाने को दो।",
    "ट्रेन कितने बजे आएगी?",
    "मेरे पास समय नहीं है, मुझे जाना होगा।",
    "बच्चे मैदान में खेल रहे हैं।",
    "यह जगह बहुत खूबसूरत है।",
    "आपसे मिलकर बहुत अच्छा लगा।",
    "क्या मैं यहाँ बैठ सकता हूँ?"
]

def get_voices():
    response = requests.get("https://api.elevenlabs.io/v1/voices", headers=HEADERS)
    if response.status_code == 200:
        voices = response.json().get('voices', [])
        return [v['voice_id'] for v in voices[:5]]
    else:
        print("Error fetching voices:", response.text)
        return []

def generate_audio(text, voice_id, output_path):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    
    response = requests.post(url, json=payload, headers=HEADERS)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        return True
    else:
        print(f"Error generating audio for {voice_id}: {response.text}")
        return False

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'data', 'raw_fake', 'fake')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Fetching available voices...")
    voices = get_voices()
    if not voices:
        print("No voices found, falling back to default voice ID.")
        voices = ["21m00Tcm4TlvDq8ikWAM"] # Rachel
        
    print(f"Generating audio using {len(voices)} voices...")
    
    count = 0
    target = 100
    
    for i in range(target):
        text = HINDI_SENTENCES[i % len(HINDI_SENTENCES)]
        voice_id = voices[i % len(voices)]
        
        output_path = os.path.join(output_dir, f"elevenlabs_hindi_{i:03d}.mp3")
        
        if os.path.exists(output_path):
            print(f"Skipping {output_path}, already exists.")
            count += 1
            continue
            
        print(f"Generating {i+1}/{target}...")
        success = generate_audio(text, voice_id, output_path)
        
        if success:
            count += 1
            
        time.sleep(1) # Simple rate limiting
        
    print(f"Successfully generated {count} ElevenLabs Hindi clips!")

if __name__ == "__main__":
    main()
