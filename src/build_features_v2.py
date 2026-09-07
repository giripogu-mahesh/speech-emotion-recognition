from pathlib import Path

import librosa
import numpy as np
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

METADATA_FILE = PROJECT_DIR / "metadata.csv"

FEATURES_DIR = PROJECT_DIR / "features_v2"

FEATURES_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Configuration
# ============================================================

SAMPLE_RATE = 16000

N_MFCC = 13

N_MELS = 40

N_FFT = 1024

HOP_LENGTH = 256

MAX_FRAMES = 250


# ============================================================
# 3. Load metadata
# ============================================================

df = pd.read_csv(METADATA_FILE)

print("Total files:", len(df))


# ============================================================
# 4. Speaker-independent split
# ============================================================

TRAIN_ACTORS = list(range(1, 17))

VAL_ACTORS = list(range(17, 21))

TEST_ACTORS = list(range(21, 25))


train_df = df[df["actor"].isin(TRAIN_ACTORS)].copy()

val_df = df[df["actor"].isin(VAL_ACTORS)].copy()

test_df = df[df["actor"].isin(TEST_ACTORS)].copy()


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
# 5. Extract MFCC + Delta + Delta-Delta
# ============================================================

def extract_features(audio_path):

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE
    )


    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )


    # --------------------------------------------------------
    # Delta MFCC
    # --------------------------------------------------------

    delta = librosa.feature.delta(
        mfcc
    )


    # --------------------------------------------------------
    # Delta-Delta MFCC
    # --------------------------------------------------------

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )


    # --------------------------------------------------------
    # Combine
    #
    # MFCC      = 13
    # Delta     = 13
    # Delta²    = 13
    #
    # Total     = 39
    # --------------------------------------------------------

    features = np.concatenate(
        [
            mfcc,
            delta,
            delta2
        ],
        axis=0
    )


    # --------------------------------------------------------
    # Fix number of time frames
    # --------------------------------------------------------

    current_frames = features.shape[1]


    if current_frames < MAX_FRAMES:

        padding = MAX_FRAMES - current_frames

        features = np.pad(
            features,
            (
                (0, 0),
                (0, padding)
            ),
            mode="edge"
        )


    elif current_frames > MAX_FRAMES:

        features = features[:, :MAX_FRAMES]


    return features.astype(np.float32)


# ============================================================
# 6. Process dataframe
# ============================================================

def process_dataframe(dataframe, name):

    features = []

    labels = []

    actors = []


    total = len(dataframe)


    for i, (_, row) in enumerate(
        dataframe.iterrows()
    ):

        feature = extract_features(
            row["filepath"]
        )


        features.append(feature)

        labels.append(
            int(row["emotion_id"])
        )

        actors.append(
            int(row["actor"])
        )


        if (
            (i + 1) % 100 == 0
            or i == 0
        ):

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
# 7. Extract all features
# ============================================================

print("\n==============================")
print("Extracting TRAIN features")
print("==============================")

X_train, y_train, actors_train = (
    process_dataframe(
        train_df,
        "Train"
    )
)


print("\n==============================")
print("Extracting VALIDATION features")
print("==============================")

X_val, y_val, actors_val = (
    process_dataframe(
        val_df,
        "Validation"
    )
)


print("\n==============================")
print("Extracting TEST features")
print("==============================")

X_test, y_test, actors_test = (
    process_dataframe(
        test_df,
        "Test"
    )
)


# ============================================================
# 8. Normalize
# ============================================================

# Calculate statistics ONLY from training data.

train_mean = X_train.mean()

train_std = X_train.std()


print("\nNormalization statistics")

print("Training mean:", train_mean)

print("Training std:", train_std)


train_std = max(
    train_std,
    1e-8
)


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
# 9. Save datasets
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


np.save(
    FEATURES_DIR / "mean.npy",
    np.array(train_mean)
)


np.save(
    FEATURES_DIR / "std.npy",
    np.array(train_std)
)


# ============================================================
# 10. Save metadata splits
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
# 11. Final output
# ============================================================

print("\n===================================")
print("VERSION 2 FEATURE EXTRACTION DONE")
print("===================================")

print("Train:", X_train.shape)

print("Validation:", X_val.shape)

print("Test:", X_test.shape)

print("\nFeatures saved to:")

print(FEATURES_DIR)