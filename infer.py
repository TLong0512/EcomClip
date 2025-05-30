import os
import time
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from utils.ecom_clip import EcomClip
from utils.load_dataset import LoadDataset


def load_model(weight_path="ecom_clip.pth", num_classes=10, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EcomClip(num_classes=num_classes, device=device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(model, dataset, weight_path="ecom_clip.pth", save_dir="visualizer"):
    os.makedirs(save_dir, exist_ok=True)

    y_true = []
    y_pred = []
    device = model.device

    # --- Start timing inference ---
    start_time = time.time()

    for image_tensor, label, _, _, in tqdm(dataset, desc="Evaluating"):
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).item()

        y_true.append(label)
        y_pred.append(pred)

    end_time = time.time()
    inference_time = (end_time - start_time) / len(dataset)

    # --- Model size ---
    model_size_mb = os.path.getsize(weight_path) / (1024 * 1024)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    # --- Classification Report ---
    report_txt = classification_report(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    class_metrics = report_df.iloc[:-3][["precision", "recall", "f1-score"]]

    # Save bar chart
    plt.figure(figsize=(12, 6))
    class_metrics.plot(kind="bar", ylim=(0, 1), colormap="Set2")
    plt.title("Classification Report Metrics per Class")
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "classification_report_chart.png"))
    plt.close()

    # Save report to TXT
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report_txt)
        f.write(f"\n\nInference time per image: {inference_time:.4f} seconds")
        f.write(f"\nModel size: {model_size_mb:.2f} MB")

    print(f"Saved confusion matrix, classification report and stats in '{save_dir}'")


if __name__ == "__main__":
    model = load_model("ecom_clip.pth", num_classes=10)
    val_dataset = LoadDataset(split="val")
    evaluate_model(model, val_dataset, weight_path="ecom_clip.pth")
