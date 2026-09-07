from pathlib import Path
import argparse

import librosa
import numpy as np
import torch
import torch.nn as nn

# ============================================================
# 1. Configuration and paths
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "best_speech_emotion_v2.pth"
FEATURES_DIR = PROJECT_DIR / "features_v2"

SAMPLE_RATE = 16000
N_MFCC = 13
N_MELS = 40
N_FFT = 1024
HOP_LENGTH = 256
MAX_FRAMES = 250

CLASS_NAMES = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. Architecture definition
# ============================================================
class SpeechEmotionCNNBiLSTM(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        self.lstm = nn.LSTM(
            input_size=128 * 4,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 2, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        batch_size, channels, frequency, time = x.shape
        x = x.permute(0, 3, 1, 2).reshape(
            batch_size, time, channels * frequency
        )
        x, _ = self.lstm(x)
        return self.classifier(x[:, -1, :])


def extract_features(audio_path):
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0)

    if features.shape[1] < MAX_FRAMES:
        features = np.pad(
            features,
            ((0, 0), (0, MAX_FRAMES - features.shape[1])),
            mode="edge"
        )
    else:
        features = features[:, :MAX_FRAMES]

    mean = np.load(FEATURES_DIR / "mean.npy")
    std = np.load(FEATURES_DIR / "std.npy")
    return ((features - mean) / max(float(std), 1e-8)).astype(np.float32)

# ============================================================
# 3. Load Model
# ============================================================
print("\nLoading locally trained CNN-BiLSTM emotion model...")
model = SpeechEmotionCNNBiLSTM(num_classes=8).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print("Model ready!\n")

# ============================================================
# 4. Predict Function
# ============================================================
def predict_emotion(audio_path):
    audio_path = Path(str(audio_path).strip("'\""))
    if not audio_path.exists():
        print(f"[Error] Audio file not found at: {audio_path}")
        return

    features = extract_features(audio_path)
    inputs = torch.from_numpy(features).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = probs.argmax()

    print("=" * 50)
    print(f"Audio File: {audio_path.name}")
    print(f"PREDICTED EMOTION: {CLASS_NAMES[pred_idx].upper()} ({probs[pred_idx]*100:.2f}% confidence)")
    print("=" * 50)
    print("Class Probabilities:")
    for name, p in zip(CLASS_NAMES, probs):
        bar = "#" * int(p * 20)
        print(f"  {name.capitalize():<10} : {p*100:5.1f}% | {bar}")
    print("=" * 50)


# ============================================================
# 5. TEST YOUR AUDIO ADDRESS HERE
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict the emotion in a speech audio file."
    )
    parser.add_argument(
        "audio_file",
        help="Path to a WAV or other librosa-supported audio file"
    )
    args = parser.parse_args()
    predict_emotion(args.audio_file)