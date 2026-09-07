from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. Find project directory
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

METADATA_FILE = PROJECT_DIR / "metadata.csv"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Load metadata
# ============================================================

df = pd.read_csv(METADATA_FILE)

print("Total files:", len(df))


# ============================================================
# 3. Select one audio file
# ============================================================

sample = df.iloc[100]

audio_path = Path(sample["filepath"])

print("\nSelected audio:")
print(audio_path)

print("\nActor:", sample["actor"])
print("Emotion:", sample["emotion"])
print("Emotion ID:", sample["emotion_id"])


# ============================================================
# 4. Load audio
# ============================================================

audio, sample_rate = librosa.load(
    audio_path,
    sr=None
)


# ============================================================
# 5. Basic audio information
# ============================================================

duration = len(audio) / sample_rate

print("\nAudio information")
print("-----------------")

print("Sampling rate:", sample_rate, "Hz")
print("Number of samples:", len(audio))
print("Duration:", round(duration, 3), "seconds")
print("Minimum amplitude:", np.min(audio))
print("Maximum amplitude:", np.max(audio))


# ============================================================
# 6. Plot waveform
# ============================================================

plt.figure(figsize=(12, 4))

librosa.display.waveshow(
    audio,
    sr=sample_rate
)

plt.title(
    f"Waveform - {sample['emotion']}"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "waveform.png",
    dpi=150
)

plt.show()


# ============================================================
# 7. Create Mel-spectrogram
# ============================================================

mel_spectrogram = librosa.feature.melspectrogram(
    y=audio,
    sr=sample_rate,
    n_fft=1024,
    hop_length=256,
    n_mels=40
)


# Convert power to decibels

mel_db = librosa.power_to_db(
    mel_spectrogram,
    ref=np.max
)


# ============================================================
# 8. Plot Mel-spectrogram
# ============================================================

plt.figure(figsize=(12, 5))

librosa.display.specshow(
    mel_db,
    sr=sample_rate,
    hop_length=256,
    x_axis="time",
    y_axis="mel"
)

plt.colorbar(format="%+2.0f dB")

plt.title(
    f"Mel-Spectrogram - {sample['emotion']}"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "mel_spectrogram.png",
    dpi=150
)

plt.show()


# ============================================================
# 9. Calculate MFCC
# ============================================================

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=13,
    n_fft=1024,
    hop_length=256,
    n_mels=40
)


# ============================================================
# 10. MFCC information
# ============================================================

print("\nMFCC information")
print("-----------------")

print("MFCC shape:", mfcc.shape)

print(
    "Number of MFCC coefficients:",
    mfcc.shape[0]
)

print(
    "Number of time frames:",
    mfcc.shape[1]
)


# ============================================================
# 11. Plot MFCC
# ============================================================

plt.figure(figsize=(12, 5))

librosa.display.specshow(
    mfcc,
    sr=sample_rate,
    hop_length=256,
    x_axis="time"
)

plt.colorbar()

plt.ylabel("MFCC coefficient")

plt.title(
    f"MFCC - {sample['emotion']}"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "mfcc.png",
    dpi=150
)

plt.show()


# ============================================================
# 12. Print part of MFCC matrix
# ============================================================

print("\nFirst 5 MFCC coefficients")
print("-------------------------")

print(mfcc[:5, :10])