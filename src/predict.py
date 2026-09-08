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
MODEL_PATH = PROJECT_DIR / "models" / "main_speech_emotion_model.pth"
FEATURES_DIR = PROJECT_DIR / "features_v3c"

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

def select_device():
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        architecture = f"sm_{capability[0]}{capability[1]}"
        if architecture in torch.cuda.get_arch_list():
            return torch.device("cuda")
        print(
            f"CUDA architecture {architecture} is not supported by this "
            "PyTorch build; using CPU."
        )
    return torch.device("cpu")


device = select_device()

# ============================================================
# 2. Architecture definition
# ============================================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        weights = torch.softmax(self.attention(x), dim=1)
        return torch.sum(x * weights, dim=1)


class EmotionCNNAttentiveV3D(nn.Module):
    def __init__(self, in_channels=45, num_classes=8, dropout=0.35):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout + 0.05)
        )
        self.attn_pool = TemporalAttention(hidden_dim=256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(self.attn_pool(x))


def extract_features(audio_path):
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    audio, _ = librosa.effects.trim(audio, top_db=25)
    if len(audio) < SAMPLE_RATE // 2:
        audio = np.pad(audio, (0, SAMPLE_RATE // 2 - len(audio)))

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
    rms = librosa.feature.rms(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )
    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    f0, _, voiced_prob = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=SAMPLE_RATE,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )
    f0 = np.nan_to_num(f0, nan=0.0)[None, :]
    voiced_prob = np.nan_to_num(voiced_prob, nan=0.0)[None, :]
    features = np.concatenate(
        [mfcc, delta, delta2, rms, zcr, centroid, rolloff, f0, voiced_prob],
        axis=0
    )

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
    normalized = (features - mean) / np.maximum(std, 1e-8)
    return np.squeeze(normalized, axis=0).astype(np.float32)

# ============================================================
# 3. Load Model
# ============================================================
print("\nLoading locally trained V3D emotion model...")
model = EmotionCNNAttentiveV3D(num_classes=8).to(device)
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
    inputs = torch.from_numpy(features).unsqueeze(0).to(device)
    
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