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

FEATURES_DIR = PROJECT_DIR / "features_v2"

MODELS_DIR = PROJECT_DIR / "models"

RESULTS_DIR = PROJECT_DIR / "results"

MODELS_DIR.mkdir(exist_ok=True)

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Configuration
# ============================================================

BATCH_SIZE = 32

EPOCHS = 50

LEARNING_RATE = 0.0005

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
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("Device:", device)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 4. Load data
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
# 5. Dataset
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

        # 39 × 250
        #
        # becomes
        #
        # 1 × 39 × 250

        x = x.unsqueeze(0)

        return x, y


# ============================================================
# 6. Datasets
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
# 7. DataLoaders
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
# 8. CNN + BiLSTM
# ============================================================

class SpeechEmotionCNNBiLSTM(
    nn.Module
):

    def __init__(self):

        super().__init__()


        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        self.cnn = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=(2, 2)
            ),


            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=(2, 2)
            ),


            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=(2, 1)
            )
        )


        # ----------------------------------------------------
        # After CNN:
        #
        # Input:
        # 1 × 39 × 250
        #
        # After pools approximately:
        #
        # 128 × 9 × 62
        #
        # We treat TIME as the sequence dimension.
        # ----------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=128 * 4,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )


        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Dropout(0.5),

            nn.Linear(
                128 * 2,
                NUM_CLASSES
            )
        )


    def forward(self, x):

        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        x = self.cnn(x)


        # x:
        #
        # batch × channels × frequency × time
        #
        # We want:
        #
        # batch × time × features

        batch_size = x.size(0)

        channels = x.size(1)

        frequency = x.size(2)

        time = x.size(3)


        x = x.permute(
            0,
            3,
            1,
            2
        )


        x = x.reshape(
            batch_size,
            time,
            channels * frequency
        )


        # ----------------------------------------------------
        # BiLSTM
        # ----------------------------------------------------

        x, _ = self.lstm(x)


        # ----------------------------------------------------
        # Use final time step
        # ----------------------------------------------------

        x = x[:, -1, :]


        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        x = self.classifier(x)


        return x


# ============================================================
# 9. Create model
# ============================================================

model = SpeechEmotionCNNBiLSTM().to(device)


print("\nModel:")

print(model)


# ============================================================
# 10. Class weights
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

for i, name in enumerate(
    CLASS_NAMES
):

    print(
        name,
        ":",
        class_counts[i]
    )


# ============================================================
# 11. Loss
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)


# ============================================================
# 12. Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# 13. Scheduler
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5
)


# ============================================================
# 14. Training
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0

    all_predictions = []

    all_labels = []


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


        # Prevent exploding gradients

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()


        total_loss += (
            loss.item()
            * X.size(0)
        )


        predictions = outputs.argmax(
            dim=1
        )


        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            y.detach()
            .cpu()
            .numpy()
        )


    loss = (
        total_loss
        /
        len(train_dataset)
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    return loss, accuracy


# ============================================================
# 15. Evaluation
# ============================================================

def evaluate(loader):

    model.eval()

    total_loss = 0

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


            all_predictions.extend(
                predictions.cpu()
                .numpy()
            )

            all_labels.extend(
                y.cpu()
                .numpy()
            )


    loss = (
        total_loss
        /
        len(loader.dataset)
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    return (
        loss,
        accuracy,
        np.array(all_labels),
        np.array(all_predictions)
    )


# ============================================================
# 16. Training loop
# ============================================================

best_val_accuracy = 0.0

best_model_path = (
    MODELS_DIR
    /
    "best_speech_emotion_v2.pth"
)


print("\n==============================")

print("STARTING VERSION 2 TRAINING")

print("==============================")


for epoch in range(EPOCHS):


    train_loss, train_accuracy = (
        train_one_epoch()
    )


    (
        val_loss,
        val_accuracy,
        _,
        _
    ) = evaluate(
        val_loader
    )


    scheduler.step(
        val_accuracy
    )


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f} | "
        f"LR: {current_lr:.6f}"
    )


    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy


        torch.save(
            model.state_dict(),
            best_model_path
        )


        print(
            "  → Best V2 model saved!"
        )


# ============================================================
# 17. Load best model
# ============================================================

print("\nLoading best V2 model...")


model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=device
    )
)


# ============================================================
# 18. Test
# ============================================================

(
    test_loss,
    test_accuracy,
    y_true,
    y_pred
) = evaluate(
    test_loader
)


test_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)


print("\n==============================")

print("VERSION 2 TEST RESULTS")

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
    "Test F1:",
    round(test_f1, 4)
)


# ============================================================
# 19. Classification report
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
# 20. Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


print("\nConfusion Matrix")

print("================")


print(cm)


# ============================================================
# 21. Save results
# ============================================================

np.savetxt(
    RESULTS_DIR
    /
    "confusion_matrix_v2.csv",
    cm,
    delimiter=",",
    fmt="%d"
)


# ============================================================
# 22. Save model
# ============================================================

final_model_path = (
    MODELS_DIR
    /
    "speech_emotion_v2_final.pth"
)


torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "class_names":
            CLASS_NAMES,

        "n_mfcc":
            N_MFCC if False else 13,

        "feature_type":
            "MFCC + Delta + Delta-Delta",

        "sample_rate":
            16000,

        "n_fft":
            1024,

        "hop_length":
            256,

        "max_frames":
            250,

        "test_accuracy":
            test_accuracy,

        "test_f1":
            test_f1
    },
    final_model_path
)


print("\nV2 model saved to:")

print(final_model_path)


print("\nVersion 2 training complete!")