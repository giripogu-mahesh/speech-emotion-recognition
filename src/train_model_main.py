from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# ============================================================
# 1. Paths & Configuration
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_DIR / "features_v3c"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 80
PATIENCE = 20
LEARNING_RATE = 0.0003
NUM_CLASSES = 8

CLASS_NAMES = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 2. Load Datasets
# ============================================================
train_data = np.load(FEATURES_DIR / "train_features.npz")
val_data   = np.load(FEATURES_DIR / "validation_features.npz")
test_data  = np.load(FEATURES_DIR / "test_features.npz")

X_train, y_train = train_data["X"], train_data["y"]
X_val, y_val     = val_data["X"], val_data["y"]
X_test, y_test   = test_data["X"], test_data["y"]

class SpeechDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

train_loader = DataLoader(SpeechDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(SpeechDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(SpeechDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 3. Temporal Self-Attention Module
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
        # x shape: (Batch, Hidden_dim, Time) -> permute to (Batch, Time, Hidden_dim)
        x_proj = x.permute(0, 2, 1)
        weights = self.attention(x_proj)  # (Batch, Time, 1)
        weights = torch.softmax(weights, dim=1)
        # Weighted temporal sum
        context = torch.sum(x_proj * weights, dim=1)  # (Batch, Hidden_dim)
        return context

# ============================================================
# 4. V3-D Final Architecture
# ============================================================
class EmotionCNN_Attentive_V3D(nn.Module):
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
        x = self.attn_pool(x)  # Replaces uniform GAP with learnable attention
        return self.classifier(x)

model = EmotionCNN_Attentive_V3D().to(device)

class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
class_weights = len(y_train) / (NUM_CLASSES * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

# ============================================================
# 5. Training Loop
# ============================================================
def run_epoch(loader, is_train=True):
    model.train() if is_train else model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    
    with torch.set_grad_enabled(is_train):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            total_loss += loss.item() * X.size(0)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.detach().cpu().numpy())
            
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, np.array(all_labels), np.array(all_preds)

best_val_acc = 0.0
patience_counter = 0
best_model_path = MODELS_DIR / "main_speech_emotion_model.pth"

print("\nStarting Training V3-D (CNN + Temporal Attention Pooling)...")

for epoch in range(EPOCHS):
    train_loss, train_acc, _, _ = run_epoch(train_loader, is_train=True)
    val_loss, val_acc, _, _     = run_epoch(val_loader, is_train=False)
    
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]
    
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"  → Best V3-D checkpoint saved! (New Best Val Acc: {best_val_acc:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

# ============================================================
# 6. Final Test Set Evaluation
# ============================================================
print("\nEvaluating best V3-D checkpoint on Test set (Actors 21-24)...")
model.load_state_dict(torch.load(best_model_path, map_location=device))
test_loss, test_acc, y_true, y_pred = run_epoch(test_loader, is_train=False)
test_f1 = f1_score(y_true, y_pred, average="weighted")

print(f"\n--- V3-D Final Test Results ---\nTest Accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

cm = confusion_matrix(y_true, y_pred)
np.savetxt(RESULTS_DIR / "confusion_matrix_main.csv", cm, delimiter=",", fmt="%d")
print("V3-D evaluation complete.")