import re
import os
import matplotlib.pyplot as plt

log_file_path = os.path.join("log", "log.txt")

output_dir = "visualizer"
os.makedirs(output_dir, exist_ok=True)

train_losses = []
val_accuracies = []
precisions = []
recalls = []
f1_scores = []

with open(log_file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(len(lines)):
    line = lines[i].strip()

    if line.startswith("Train Loss:"):
        loss = float(line.split(":")[1].strip())
        train_losses.append(loss)

    if line.startswith("Validation Accuracy:"):
        acc = float(line.split(":")[1].replace("%", "").strip()) / 100
        val_accuracies.append(acc)

    if "precision" in line and "recall" in line and "f1-score" in line:
        if i + 1 < len(lines):
            values_line = lines[i + 1].strip()
            values = re.split(r'\s+', values_line)
            if len(values) >= 4:
                precisions.append(float(values[1]))
                recalls.append(float(values[2]))
                f1_scores.append(float(values[3]))

epochs = list(range(1, len(train_losses) + 1))

def save_plot(x, y_values, labels, title, ylabel, filename):
    plt.figure(figsize=(10, 5))
    for y, label in zip(y_values, labels):
        plt.plot(x, y, marker='o', label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

save_plot(epochs, [train_losses], ["Loss"], "Training Loss Over Epochs", "Loss", "loss.png")

save_plot(epochs, [val_accuracies], ["Accuracy"], "Validation Accuracy Over Epochs", "Accuracy", "accuracy.png")

save_plot(epochs, [precisions, recalls, f1_scores],
          ["Precision", "Recall", "F1-score"],
          "Precision, Recall, F1-score Over Epochs",
          "Score",
          "prf_metrics.png")
