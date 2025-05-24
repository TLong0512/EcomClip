from torch.utils.data import DataLoader
from load_dataset import LoadDataset

train_dataset = LoadDataset(split="train")
val_dataset = LoadDataset(split="val")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



for images, labels, descriptions, paths in train_loader:
    print(images.shape)        # torch.Size([32, 3, 224, 224])
    print(labels.shape)        # torch.Size([32])
    print(descriptions[0])     # Ví dụ: "áo dài tay nữ"
    print(paths[0])            # Đường dẫn ảnh đầy đủ
    break
