from pathlib import Path

import librosa
import numpy as np
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

METADATA_FILE = PROJECT_DIR / "metadata.csv"

FEATURES_DIR = PROJECT_DIR / "features"

FEATURES_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Configuration
# ============================================================

SAMPLE_RATE = 16000

N_MFCC = 13

N_MELS = 40

N_FFT = 1024

HOP_LENGTH = 256

# All MFCC matrices will become:
#
# 13 × 250
#
# If an audio file has fewer than 250 frames,
# we pad it with zeros.
#
# If it has more than 250 frames,
# we truncate it.

MAX_FRAMES = 250


# ============================================================
# 3. Load metadata
# ============================================================

df = pd.read_csv(METADATA_FILE)

print("Total files:", len(df))


# ============================================================
# 4. Speaker-independent split
# ============================================================

TRAIN_ACTORS = list(range(1, 17))      # 1-16

VAL_ACTORS = list(range(17, 21))       # 17-20

TEST_ACTORS = list(range(21, 25))      # 21-24


train_df = df[df["actor"].isin(TRAIN_ACTORS)].copy()

val_df = df[df["actor"].isin(VAL_ACTORS)].copy()

test_df = df[df["actor"].isin(TEST_ACTORS)].copy()


# ============================================================
# 5. Verify speaker split
# ============================================================

print("\nSpeaker split")
print("-------------")

print(
    "Train actors:",
    sorted(train_df["actor"].unique())
)

print(
    "Validation actors:",
    sorted(val_df["actor"].unique())
)

print(
    "Test actors:",
    sorted(test_df["actor"].unique())
)


print("\nNumber of audio files")

print("Train:", len(train_df))

print("Validation:", len(val_df))

print("Test:", len(test_df))


# ============================================================
# 6. Save split metadata
# ============================================================

train_df.to_csv(
    FEATURES_DIR / "train.csv",
    index=False
)

val_df.to_csv(
    FEATURES_DIR / "validation.csv",
    index=False
)

test_df.to_csv(
    FEATURES_DIR / "test.csv",
    index=False
)


# ============================================================
# 7. MFCC extraction function
# ============================================================

def extract_mfcc(audio_path):

    # Load audio at exactly 16 kHz

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE
    )


    # Calculate MFCC

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )


    # --------------------------------------------------------
    # Make every sample the same size
    # --------------------------------------------------------

    current_frames = mfcc.shape[1]


    if current_frames < MAX_FRAMES:

        # Number of frames that need to be added

        padding = MAX_FRAMES - current_frames

        mfcc = np.pad(
            mfcc,
            (
                (0, 0),
                (0, padding)
            ),
            mode="constant"
        )


    elif current_frames > MAX_FRAMES:

        # Keep only the first MAX_FRAMES

        mfcc = mfcc[:, :MAX_FRAMES]


    return mfcc.astype(np.float32)


# ============================================================
# 8. Extract features for a dataframe
# ============================================================

def process_dataframe(dataframe, name):

    features = []

    labels = []

    actors = []


    total = len(dataframe)


    for i, (_, row) in enumerate(dataframe.iterrows()):

        audio_path = row["filepath"]


        mfcc = extract_mfcc(audio_path)


        features.append(mfcc)

        labels.append(row["emotion_id"])

        actors.append(row["actor"])


        # Progress

        if (i + 1) % 100 == 0 or i == 0:

            print(
                f"{name}: "
                f"{i + 1}/{total} files processed"
            )


    features = np.stack(features)

    labels = np.array(
        labels,
        dtype=np.int64
    )

    actors = np.array(
        actors,
        dtype=np.int64
    )


    print(
        f"\n{name} feature shape:",
        features.shape
    )


    return features, labels, actors


# ============================================================
# 9. Extract TRAIN features
# ============================================================

print("\n==============================")
print("Extracting TRAIN features")
print("==============================")

X_train, y_train, actors_train = process_dataframe(
    train_df,
    "Train"
)


# ============================================================
# 10. Extract VALIDATION features
# ============================================================

print("\n==============================")
print("Extracting VALIDATION features")
print("==============================")

X_val, y_val, actors_val = process_dataframe(
    val_df,
    "Validation"
)


# ============================================================
# 11. Extract TEST features
# ============================================================

print("\n==============================")
print("Extracting TEST features")
print("==============================")

X_test, y_test, actors_test = process_dataframe(
    test_df,
    "Test"
)


# ============================================================
# 12. Normalize features
# ============================================================

# IMPORTANT:
#
# Calculate mean and standard deviation ONLY from
# training data.
#
# We must NOT use validation/test information here.

train_mean = X_train.mean()

train_std = X_train.std()

print("\nNormalization statistics")

print("Training mean:", train_mean)

print("Training std:", train_std)


# Avoid division by zero

train_std = max(train_std, 1e-8)


X_train = (
    X_train - train_mean
) / train_std


X_val = (
    X_val - train_mean
) / train_std


X_test = (
    X_test - train_mean
) / train_std


# ============================================================
# 13. Save everything
# ============================================================

np.savez_compressed(
    FEATURES_DIR / "train_features.npz",
    X=X_train,
    y=y_train,
    actors=actors_train
)


np.savez_compressed(
    FEATURES_DIR / "validation_features.npz",
    X=X_val,
    y=y_val,
    actors=actors_val
)


np.savez_compressed(
    FEATURES_DIR / "test_features.npz",
    X=X_test,
    y=y_test,
    actors=actors_test
)


# Save normalization statistics

np.save(
    FEATURES_DIR / "mean.npy",
    np.array(train_mean)
)

np.save(
    FEATURES_DIR / "std.npy",
    np.array(train_std)
)


# ============================================================
# 14. Final information
# ============================================================

print("\n===================================")
print("FEATURE EXTRACTION COMPLETE")
print("===================================")

print("\nTrain:", X_train.shape)

print("Validation:", X_val.shape)

print("Test:", X_test.shape)

print("\nFeatures saved inside:")

print(FEATURES_DIR)