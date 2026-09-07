from pathlib import Path
import librosa
import numpy as np
import pandas as pd

# ============================================================
# 1. Paths & Configuration
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
METADATA_FILE = PROJECT_DIR / "metadata.csv"
FEATURES_DIR = PROJECT_DIR / "features_v3c"
FEATURES_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000
N_MFCC = 13
N_MELS = 40
N_FFT = 1024
HOP_LENGTH = 256
MAX_FRAMES = 250

df = pd.read_csv(METADATA_FILE)

TRAIN_ACTORS = list(range(1, 17))
VAL_ACTORS = list(range(17, 21))
TEST_ACTORS = list(range(21, 25))

train_df = df[df["actor"].isin(TRAIN_ACTORS)].copy()
val_df = df[df["actor"].isin(VAL_ACTORS)].copy()
test_df = df[df["actor"].isin(TEST_ACTORS)].copy()

# ============================================================
# 2. Rich Multi-Acoustic Block (45 Channels)
# ============================================================
def compute_acoustic_block(audio):
    # 1. MFCC + Delta + Delta-Delta (39 channels)
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # 2. RMS Energy (1 channel)
    rms = librosa.feature.rms(y=audio, frame_length=N_FFT, hop_length=HOP_LENGTH)
    
    # 3. Zero Crossing Rate (1 channel)
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=N_FFT, hop_length=HOP_LENGTH)
    
    # 4. Spectral Centroid & Rolloff (2 channels)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    # 5. Fundamental Frequency F0 / Pitch (2 channels: F0 and voiced probability)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), 
        sr=SAMPLE_RATE, frame_length=N_FFT, hop_length=HOP_LENGTH
    )
    # Replace unvoiced NaNs with 0
    f0 = np.nan_to_num(f0, nan=0.0)
    voiced_prob = np.nan_to_num(voiced_prob, nan=0.0)
    f0 = np.expand_dims(f0, axis=0)
    voiced_prob = np.expand_dims(voiced_prob, axis=0)
    
    # Combine: 13 + 13 + 13 + 1 + 1 + 1 + 1 + 1 + 1 = 45 channels
    features = np.concatenate([mfcc, delta, delta2, rms, zcr, centroid, rolloff, f0, voiced_prob], axis=0)
    
    # Fix time frame length
    current_frames = features.shape[1]
    if current_frames < MAX_FRAMES:
        features = np.pad(features, ((0, 0), (0, MAX_FRAMES - current_frames)), mode="edge")
    else:
        features = features[:, :MAX_FRAMES]
        
    return features.astype(np.float32)

# ============================================================
# 3. Augmentations
# ============================================================
def extract_features(audio_path, augment=False):
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    audio, _ = librosa.effects.trim(audio, top_db=25)
    
    if len(audio) < SAMPLE_RATE // 2:
        audio = np.pad(audio, (0, (SAMPLE_RATE // 2) - len(audio)))

    if not augment:
        return [compute_acoustic_block(audio)]
    
    variants = []
    
    # 1. Original
    variants.append(compute_acoustic_block(audio))
    
    # 2. Additive Noise
    noise = np.random.randn(len(audio))
    noisy_audio = audio + 0.004 * noise
    variants.append(compute_acoustic_block(noisy_audio.astype(np.float32)))
    
    # 3. Pitch Shift
    shift = np.random.choice([-2.0, 2.0])
    try:
        shifted_audio = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=shift)
        variants.append(compute_acoustic_block(shifted_audio))
    except Exception:
        variants.append(compute_acoustic_block(audio))
    
    # 4. Speed Resampling
    rate = np.random.choice([0.9, 1.1])
    try:
        orig_len = len(audio)
        speed_audio = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=int(SAMPLE_RATE * rate))
        if len(speed_audio) > orig_len:
            speed_audio = speed_audio[:orig_len]
        else:
            speed_audio = np.pad(speed_audio, (0, orig_len - len(speed_audio)))
        variants.append(compute_acoustic_block(speed_audio.astype(np.float32)))
    except Exception:
        variants.append(compute_acoustic_block(audio))
        
    return variants

def process_dataframe(dataframe, name, is_train=False):
    features, labels, actors = [], [], []
    total = len(dataframe)
    for i, (_, row) in enumerate(dataframe.iterrows()):
        variants = extract_features(row["filepath"], augment=is_train)
        for variant in variants:
            features.append(variant)
            labels.append(int(row["emotion_id"]))
            actors.append(int(row["actor"]))
        if (i + 1) % 100 == 0 or i == 0:
            print(f"{name}: {i + 1}/{total} files processed")
    return np.stack(features), np.array(labels, dtype=np.int64), np.array(actors, dtype=np.int64)

print("\nExtracting Train (4x Augmentation + 45 Features)...")
X_train, y_train, actors_train = process_dataframe(train_df, "Train", is_train=True)

print("\nExtracting Validation (Clean)...")
X_val, y_val, actors_val = process_dataframe(val_df, "Validation", is_train=False)

print("\nExtracting Test (Clean)...")
X_test, y_test, actors_test = process_dataframe(test_df, "Test", is_train=False)

# Per-channel normalization across 45 channels
train_mean = np.mean(X_train, axis=(0, 2), keepdims=True)
train_std = np.std(X_train, axis=(0, 2), keepdims=True) + 1e-8

X_train = (X_train - train_mean) / train_std
X_val   = (X_val - train_mean) / train_std
X_test  = (X_test - train_mean) / train_std

np.savez_compressed(FEATURES_DIR / "train_features.npz", X=X_train, y=y_train, actors=actors_train)
np.savez_compressed(FEATURES_DIR / "validation_features.npz", X=X_val, y=y_val, actors=actors_val)
np.savez_compressed(FEATURES_DIR / "test_features.npz", X=X_test, y=y_test, actors=actors_test)
np.save(FEATURES_DIR / "mean.npy", train_mean)
np.save(FEATURES_DIR / "std.npy", train_std)

print(f"\nV3-C Feature extraction complete: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")