import os
from pathlib import Path
from PIL import Image
import pandas as pd
import yaml
from torchvision import transforms
from torch.utils.data import Dataset


class LoadDataset(Dataset):
    def __init__(self,
                 split="train", 
                 dataset_root="image_dataset/MEPC10",
                 metadata_csv="image_dataset/MEPC10/metadata-MEPC10.csv",
                 label_description_path="label_to_description.yaml"):
        """
        Dataset loader for the MEPC10 dataset. Returns (image_tensor, label, description, image_path)

        :param split: 'train' or 'val'
        :param dataset_root: Root directory of the dataset
        :param metadata_csv: Metadata file containing class information
        :param label_description_path: YAML file mapping labels to descriptions
        """
        self.split = split.lower()
        assert self.split in ["train", "val"], "split must be 'train' or 'val'"

        # Load label-to-description mapping
        with open(label_description_path, "r", encoding="utf-8") as f:
            self.label_to_description = yaml.safe_load(f)

        # Load metadata from CSV
        metadata = pd.read_csv(metadata_csv, sep='\t', encoding='utf-8')

        # Build mapping: num_class -> category_key
        self.class_map = {row['num_class']: row[self.split] for _, row in metadata.iterrows()}

        # Collect all image data from the selected split
        self.data = []
        split_dir = Path(dataset_root) / self.split
        for class_id_str, category_key in self.class_map.items():
            class_dir = split_dir / str(class_id_str)
            if not class_dir.exists():
                continue
            for img_file in class_dir.glob("*.*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    self.data.append({
                        "image_path": img_file,
                        "category_key": category_key,
                        "class_id": int(class_id_str)
                    })

        # Image transformation (as used in CLIP/ViT models)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert("RGB")
        image_tensor = self.transform(image)

        label = item['class_id']
        category_key = item['category_key']
        description = self.label_to_description.get(category_key, "No description available")
        image_path = str(item['image_path'])

        return image_tensor, label, description, image_path
