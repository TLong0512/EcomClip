import os
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

from utils.load_dataset import LoadDataset
from utils.ecom_clip import EcomClip

# Tạo thư mục log và visualizer nếu chưa có
os.makedirs("log", exist_ok=True)
os.makedirs("visualizer", exist_ok=True)

# Load dataset và DataLoader
train_dataset = LoadDataset(split="train")
val_dataset = LoadDataset(split="val")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Khởi tạo model, criterion, optimizer
model_ft = EcomClip(num_classes=10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ft.classifier.parameters(), lr=1e-4)

num_epochs = 200

# Các list lưu giá trị theo epoch
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
val_precisions = []
val_recalls = []
val_f1s = []

for epoch in range(num_epochs):
    model_ft.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Loss: 0.0000")
    for images, labels, _, _ in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_ft(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted_train = torch.max(outputs.data, 1)
        correct_train += (predicted_train == labels).sum().item()
        total_train += labels.size(0)
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    train_loss_epoch = running_loss / len(train_loader)
    train_acc_epoch = 100 * correct_train / total_train

    train_losses.append(train_loss_epoch)
    train_accuracies.append(train_acc_epoch)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_acc_epoch:.2f}%")

    # Validation
    model_ft.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _, _ in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_ft(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            _, predicted_val = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted_val == labels).sum().item()

            all_preds.extend(predicted_val.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss_epoch = val_running_loss / len(val_loader)
    val_acc_epoch = 100 * correct_val / total_val

    val_losses.append(val_loss_epoch)
    val_accuracies.append(val_acc_epoch)

    # Tính precision, recall, f1-score (macro)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    val_precisions.append(precision)
    val_recalls.append(recall)
    val_f1s.append(f1)

    print(f"Validation Loss: {val_loss_epoch:.4f}, Acc: {val_acc_epoch:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Ghi log epoch ra file json line
    log_dict = {
        "epoch": epoch + 1,
        "train_loss": train_loss_epoch,
        "train_accuracy": train_acc_epoch,
        "val_loss": val_loss_epoch,
        "val_accuracy": val_acc_epoch,
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1
    }
    with open("log/training_log.txt", "a") as f_log:
        f_log.write(json.dumps(log_dict) + "\n")

# Vẽ biểu đồ

# 1. Accuracy train-val
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy", color="blue", linewidth=2)
plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy", color="orange", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Train vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("visualizer/accuracy_train_val.png")
plt.close()

# 2. Loss train-val
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", color="blue", linewidth=2)
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", color="orange", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("visualizer/loss_train_val.png")
plt.close()

# 3. Precision, Recall, F1-score val
epochs = range(1, num_epochs + 1)

# Precision
plt.figure(figsize=(10, 6))
plt.plot(epochs, val_precisions, label="Precision", color="green", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.title("Validation Precision Over Epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig("visualizer/val_precision.png")
plt.close()

# Recall
plt.figure(figsize=(10, 6))
plt.plot(epochs, val_recalls, label="Recall", color="red", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.title("Validation Recall Over Epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig("visualizer/val_recall.png")
plt.close()

# F1-score
plt.figure(figsize=(10, 6))
plt.plot(epochs, val_f1s, label="F1-score", color="purple", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("F1-score")
plt.title("Validation F1-score Over Epochs")
plt.grid(True)
plt.tight_layout()
plt.savefig("visualizer/val_f1score.png")
plt.close()

# Lưu model cuối
torch.save(model_ft.state_dict(), 'ecom_clip.pth')
