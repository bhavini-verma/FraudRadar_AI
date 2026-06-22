# FraudRadar AI

## Digital Arrest Scam & Voice Clone Detection Platform

FraudRadar AI is an AI-powered voice fraud detection platform designed to identify AI-generated voices, replay attacks, and impersonation attempts in real time. The system helps protect citizens, banks, telecom providers, and public safety organizations from emerging voice-based fraud threats such as digital arrest scams and synthetic voice impersonation.

---

## Problem Statement

Recent advancements in AI voice cloning technology have enabled fraudsters to generate highly convincing synthetic voices using only a few seconds of audio. These cloned voices are increasingly being used in:

- Digital Arrest Scams
- Banking Fraud
- Customer Impersonation
- Replay Attacks
- Voice Authentication Bypass
- Social Engineering Attacks

Traditional security systems rely heavily on caller ID, metadata, and conventional voice biometrics, all of which can be manipulated or bypassed by modern AI-generated speech.

FraudRadar AI addresses this challenge by analyzing the authenticity of incoming audio in real time.

---

## Key Features

### AI Voice Clone Detection
Detects synthetic voices generated using modern text-to-speech and voice cloning systems.

### Replay Attack Detection
Identifies recorded and replayed audio designed to bypass voice authentication systems.

### Real-Time Risk Classification
Classifies calls as:

- 🟢 REAL
- 🟡 SUSPICIOUS
- 🔴 FAKE

within 10 seconds of call initiation.

### Dual Evidence Analysis
Combines biological voice characteristics with deep-learning-based speech representations for improved detection accuracy.

### Hindi & English Support
Optimized for Indian languages and regional accents.

### Zero Enrollment Required
Does not require pre-recorded voice samples or voiceprints for detection.

### Banking & Telecom Ready
Designed for deployment across banking systems, telecom networks, cybercrime units, and public safety infrastructures.

---

## System Architecture

Audio Input
↓
Preprocessing
↓
Biological Feature Stream
- Jitter
- Shimmer
- Harmonic-to-Noise Ratio (HNR)
- Pitch Analysis
- Replay Features

+

Deep Learning Stream
- Indic Wav2Vec2
- Deep Speech Embeddings
- Temporal Features

↓

Fusion Engine
- Risk Scoring
- Decision Logic

↓

Output
- REAL
- SUSPICIOUS
- FAKE

↓

Fraud Alert Dashboard

---

## Technology Stack

### Machine Learning & AI
- Indic Wav2Vec2
- XGBoost
- Feature Fusion
- Replay Detection Engine

### Audio Processing
- Python
- Librosa
- NumPy
- SciPy

### Interface
- Streamlit

### Hardware Deployment
- FPGA/VLSI Ready
- Verilog
- Xilinx Vivado

---

## Datasets

The project is trained and evaluated using a combination of:

- WaveFake Dataset
- ASVspoof Dataset
- Hindi Speech Samples
- AI-Generated Voice Samples
- Replay Attack Samples

---

## Use Cases

### Citizen Fraud Shield
Analyze suspicious calls and voice messages before financial loss occurs.

### Digital Arrest Scam Detection
Detect impersonation attempts and AI-generated voices commonly used in digital arrest scams.

### Banking Call Centers
Prevent fraudulent account modifications and transaction approvals.

### Mobile Banking Voice Authentication
Verify that voice logins originate from genuine human speakers.

### KYC & Account Opening
Detect synthetic media and fraudulent voice submissions.

### Fraud Investigation
Support post-call forensic analysis and fraud evidence generation.

---

## Business Impact

- Reduced Fraud Losses
- Citizen Scam Protection
- Faster Fraud Investigation
- Enhanced Banking & Telecom Security
- Improved Trust in Voice-Based Systems

---

## Future Scope

- Multilingual Support
- WhatsApp Integration
- IVR Integration
- Mobile Application Deployment
- FPGA Acceleration
- Enterprise Deployment
- Telecom Network Integration
- Public Safety Deployment

---

## Alignment with Problem Statement 6

FraudRadar AI directly addresses:

- AI Voice Detection
- Voice Spoof Detection
- Digital Arrest Scam Detection
- Citizen Fraud Protection
- Real-Time Fraud Alerting
- Banking & Telecom Security
- Speech AI for Public Safety

---

## Team

**Bhavini Verma**  
VIT Vellore

---

## Tagline

**Detecting Voice Fraud Before Financial Loss Occurs**