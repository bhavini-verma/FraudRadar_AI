"""
Downloads a small set of real + fake audio samples to get started.
Real: LJ Speech samples (public domain)
Fake: WaveFake subset (small, free)
"""

import os, urllib.request, zipfile
from tqdm import tqdm

os.makedirs("data/real", exist_ok=True)
os.makedirs("data/fake", exist_ok=True)

# WaveFake small subset — openly available on GitHub
WAVEFAKE_URL = "https://github.com/RUB-SysSec/WaveFake/releases/download/v1/wavefake_small.zip"

print("Downloading WaveFake sample...")
urllib.request.urlretrieve(WAVEFAKE_URL, "wavefake_small.zip")

with zipfile.ZipFile("wavefake_small.zip", "r") as z:
    z.extractall("data/")

print("Done. Check data/real/ and data/fake/")