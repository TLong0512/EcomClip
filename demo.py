import os
import gradio as gr
import torch
from PIL import Image
import pandas as pd
import yaml

from utils.ecom_clip import EcomClip

# Setup temp dir
temp_dir = "/home/thanhlong/tmp_gradio_temp"
os.makedirs(temp_dir, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = temp_dir

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EcomClip(num_classes=10, device=device)
model.load_state_dict(torch.load("ecom_clip.pth", map_location=device))
model.eval()
preprocess = model.get_preprocess()

# Load metadata
metadata = pd.read_csv("image_dataset/MEPC10/metadata-MEPC10.csv", sep="\t", encoding="utf-8")
numclass_to_category_key = {row['num_class']: row['val'] for _, row in metadata.iterrows()}
with open("label_to_description.yaml", "r", encoding="utf-8") as f:
    label_to_description = yaml.safe_load(f)

FIXED_IMAGE_FOLDER = "demo_image"

# Global mapping: basename -> full path
image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
basename_to_fullpath = {
    fname: os.path.join(FIXED_IMAGE_FOLDER, fname)
    for fname in os.listdir(FIXED_IMAGE_FOLDER)
    if fname.lower().endswith(image_extensions)
}

# Load demo images for gallery
def load_images_fixed_folder():
    items = []
    for fname, path in basename_to_fullpath.items():
        try:
            img = Image.open(path)
            items.append((img, fname))  # fname as caption
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return items

# Predict function
def predict(image):
    if image is None:
        return "No image", "", ""
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        pred_class_id = torch.argmax(logits, dim=1).item()

    category_key = numclass_to_category_key.get(pred_class_id, "unknown")
    description = label_to_description.get(category_key, "No description available")

    level1, level2, level3 = "", "", ""
    try:
        parts = [p.strip() for p in description.split(",")]
        for part in parts:
            low = part.lower()
            if low.startswith("level 1 is"):
                level1 = part[len("Level 1 is"):].strip()
            elif low.startswith("level 2 is"):
                level2 = part[len("Level 2 is"):].strip()
            elif low.startswith("level 3 is"):
                level3 = part[len("Level 3 is"):].strip()
    except Exception as e:
        print("Parsing error:", e)
        level1 = description

    return level1, level2, level3

# Handle gallery click
def handle_gallery_click(evt: gr.SelectData):
    selected_name = evt.value[1] if isinstance(evt.value, list) and len(evt.value) == 2 else evt.value
    real_path = basename_to_fullpath.get(selected_name)
    if real_path is None:
        print("Cannot find real path for:", selected_name)
        return None
    try:
        img = Image.open(real_path)
        return img
    except Exception as e:
        print("Error loading image:", e)
        return None

# UI with Gradio
with gr.Blocks() as demo:
    gr.Markdown("<h1>Predict category by EcomClip</h1>")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Predicted Image",
                height=400,
                container=True,
                show_label=True,
                show_download_button=False,
                elem_id="image-input"
            )
        with gr.Column(scale=1):
            level_1 = gr.Textbox(label="Label level 1", interactive=False)
            level_2 = gr.Textbox(label="Label level 2", interactive=False)
            level_3 = gr.Textbox(label="Label level 3", interactive=False)
            predict_button = gr.Button("Search Category")

    with gr.Row():
        gallery = gr.Gallery(label="Sample Image").style(
            grid=6, height=200, object_fit="contain"
        )

    demo.load(fn=load_images_fixed_folder, outputs=gallery)
    gallery.select(fn=handle_gallery_click, outputs=[image_input])
    predict_button.click(fn=predict, inputs=image_input, outputs=[level_1, level_2, level_3])

if __name__ == "__main__":
    demo.launch(share=True)
