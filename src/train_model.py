from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# ============================================================
# 1. Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

FEATURES_DIR = PROJECT_DIR / "features"

MODELS_DIR = PROJECT_DIR / "models"

RESULTS_DIR = PROJECT_DIR / "results"

MODELS_DIR.mkdir(exist_ok=True)

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Configuration
# ============================================================

BATCH_SIZE = 32

EPOCHS = 40

LEARNING_RATE = 0.001

NUM_CLASSES = 8


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


# ============================================================
# 3. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 4. Load feature data
# ============================================================

train_data = np.load(
    FEATURES_DIR / "train_features.npz"
)

val_data = np.load(
    FEATURES_DIR / "validation_features.npz"
)

test_data = np.load(
    FEATURES_DIR / "test_features.npz"
)


X_train = train_data["X"]

y_train = train_data["y"]


X_val = val_data["X"]

y_val = val_data["y"]


X_test = test_data["X"]

y_test = test_data["y"]


print("\nData shapes")

print("Train:", X_train.shape)

print("Validation:", X_val.shape)

print("Test:", X_test.shape)


# ============================================================
# 5. PyTorch Dataset
# ============================================================

class SpeechDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )


    def __len__(self):

        return len(self.X)


    def __getitem__(self, index):

        x = self.X[index]

        y = self.y[index]


        # Add channel dimension
        #
        # Before:
        # 13 × 250
        #
        # After:
        # 1 × 13 × 250

        x = x.unsqueeze(0)


        return x, y


# ============================================================
# 6. Create datasets
# ============================================================

train_dataset = SpeechDataset(
    X_train,
    y_train
)

val_dataset = SpeechDataset(
    X_val,
    y_val
)

test_dataset = SpeechDataset(
    X_test,
    y_test
)


# ============================================================
# 7. Create DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 8. CNN model
# ============================================================

class SpeechEmotionCNN(nn.Module):

    def __init__(self):

        super().__init__()


        self.features = nn.Sequential(

            # Input:
            # 1 × 13 × 250

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # 32 × 6 × 125

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # 64 × 3 × 62

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),


            # Make spatial size fixed

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.4),

            nn.Linear(
                128,
                NUM_CLASSES
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


model = SpeechEmotionCNN().to(device)


print("\nModel:")
print(model)


# ============================================================
# 9. Class weights
# ============================================================

class_counts = np.bincount(
    y_train,
    minlength=NUM_CLASSES
)

class_weights = (
    len(y_train)
    /
    (
        NUM_CLASSES
        *
        class_counts
    )
)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)


print("\nClass counts:")

for i, name in enumerate(CLASS_NAMES):

    print(
        name,
        ":",
        class_counts[i]
    )


# ============================================================
# 10. Loss function
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# 11. Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 12. Training functions
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0

    correct = 0

    total = 0


    for X, y in train_loader:

        X = X.to(device)

        y = y.to(device)


        optimizer.zero_grad()


        outputs = model(X)


        loss = criterion(
            outputs,
            y
        )


        loss.backward()

        optimizer.step()


        total_loss += (
            loss.item()
            * X.size(0)
        )


        predictions = outputs.argmax(
            dim=1
        )


        correct += (
            predictions == y
        ).sum().item()


        total += X.size(0)


    average_loss = (
        total_loss / total
    )

    accuracy = (
        correct / total
    )


    return average_loss, accuracy


# ============================================================
# 13. Validation
# ============================================================

def evaluate(loader):

    model.eval()

    total_loss = 0

    correct = 0

    total = 0


    all_predictions = []

    all_labels = []


    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)

            y = y.to(device)


            outputs = model(X)


            loss = criterion(
                outputs,
                y
            )


            total_loss += (
                loss.item()
                * X.size(0)
            )


            predictions = outputs.argmax(
                dim=1
            )


            correct += (
                predictions == y
            ).sum().item()


            total += X.size(0)


            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                y.cpu().numpy()
            )


    average_loss = (
        total_loss / total
    )

    accuracy = (
        correct / total
    )


    return (
        average_loss,
        accuracy,
        np.array(all_labels),
        np.array(all_predictions)
    )


# ============================================================
# 14. Training loop
# ============================================================

best_val_accuracy = 0.0

best_model_path = (
    MODELS_DIR / "best_speech_emotion_cnn.pth"
)


print("\n==============================")
print("STARTING TRAINING")
print("==============================")


for epoch in range(EPOCHS):


    train_loss, train_accuracy = (
        train_one_epoch()
    )


    val_loss, val_accuracy, _, _ = (
        evaluate(val_loader)
    )


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )


    # Save best model

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            "  → Best model saved!"
        )


# ============================================================
# 15. Load best model
# ============================================================

print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=device
    )
)


# ============================================================
# 16. Test
# ============================================================

test_loss, test_accuracy, y_true, y_pred = (
    evaluate(test_loader)
)


test_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)


print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

print(
    "Test Loss:",
    round(test_loss, 4)
)

print(
    "Test Accuracy:",
    round(test_accuracy, 4)
)

print(
    "Test F1 Score:",
    round(test_f1, 4)
)


# ============================================================
# 17. Classification report
# ============================================================

print("\nClassification Report")
print("=====================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
)


# ============================================================
# 18. Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


print("\nConfusion Matrix")
print("================")

print(cm)


# ============================================================
# 19. Save confusion matrix
# ============================================================

np.savetxt(
    RESULTS_DIR / "confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d"
)


# ============================================================
# 20. Save final model
# ============================================================

final_model_path = (
    MODELS_DIR / "speech_emotion_cnn_final.pth"
)


torch.save(
    {
        "model_state_dict": model.state_dict(),

        "class_names": CLASS_NAMES,

        "n_mfcc": 13,

        "n_mels": 40,

        "sample_rate": 16000,

        "n_fft": 1024,

        "hop_length": 256,

        "max_frames": 250,

        "test_accuracy": test_accuracy,

        "test_f1": test_f1
    },
    final_model_path
)


print("\nFinal model saved to:")

print(final_model_path)

print("\nTraining complete!")