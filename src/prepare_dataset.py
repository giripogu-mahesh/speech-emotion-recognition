from pathlib import Path
import pandas as pd


# ==================================================
# 1. Find the project directory
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "Audio_Speech_Actors_01-24_16k"

METADATA_FILE = PROJECT_DIR / "metadata.csv"


# ==================================================
# 2. RAVDESS emotion mapping
# ===============================================

emotion_mapping = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


# Numerical labels for the neural network

emotion_to_id = {
    "neutral": 0,
    "calm": 1,
    "happy": 2,
    "sad": 3,
    "angry": 4,
    "fearful": 5,
    "disgust": 6,
    "surprised": 7
}


# ==================================================
# 3. Check that the dataset exists
# ==================================================

print("Project directory:")
print(PROJECT_DIR)

print("\nDataset directory:")
print(DATASET_DIR)

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"\nDataset folder was not found:\n{DATASET_DIR}"
    )


# ==================================================
# 4. Find all audio files
# ==================================================

records = []

for actor_folder in sorted(DATASET_DIR.glob("Actor_*")):

    # Example:
    # Actor_01 → 01

    actor_id = int(actor_folder.name.split("_")[1])

    for audio_file in sorted(actor_folder.glob("*.wav")):

        # Example filename:
        # 03-01-05-02-02-02-12.wav

        parts = audio_file.stem.split("-")

        # Third value represents emotion
        emotion_code = parts[2]

        emotion = emotion_mapping[emotion_code]

        records.append({
            "filepath": str(audio_file.relative_to(PROJECT_DIR)),
            "actor": actor_id,
            "emotion": emotion,
            "emotion_id": emotion_to_id[emotion]
        })


# ==================================================
# 5. Create DataFrame
# ==================================================

df = pd.DataFrame(records)


# ==================================================
# 6. Make sure files were found
# ==================================================

if len(df) == 0:
    raise RuntimeError(
        "\nNo .wav files were found.\n"
        "Please check the RAVDESS folder structure."
    )


# ==================================================
# 7. Display dataset information
# ==================================================

print("\nDataset information")
print("-------------------")

print("Total audio files:", len(df))
print("Number of actors:", df["actor"].nunique())

print("\nEmotion distribution:")
print(df["emotion"].value_counts().sort_index())


print("\nFirst 10 rows:")
print(df.head(10))


# ==================================================
# 8. Save metadata
# ==================================================

df.to_csv(METADATA_FILE, index=False)

print("\nmetadata.csv created successfully!")
print("Saved at:")
print(METADATA_FILE)