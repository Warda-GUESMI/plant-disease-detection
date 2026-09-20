import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from pathlib import Path

def get_train_transforms(img_size: int = 224) -> A.Compose:
    try:
        # Albumentations v2.0+
        crop = A.RandomResizedCrop(size=(img_size, img_size), scale=(0.85, 1.0), ratio=(0.9, 1.1))
    except Exception:
        # Albumentations v1.x
        crop = A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1))

    return A.Compose([
        crop,
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_REFLECT, p=0.4),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=15),
        ], p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms(img_size: int = 224) -> A.Compose:
    try:
        resize = A.Resize(size=(img_size, img_size))
    except Exception:
        resize = A.Resize(height=img_size, width=img_size)

    return A.Compose([
        resize,
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

class PlantDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.dataset = datasets.ImageFolder(root_dir)
        self.transform = transform
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']
        return image, label

def create_dataloaders(splits_dir: str, img_size: int = 224, batch_size: int = 32, num_workers: int = 4):
    splits_path = Path(splits_dir)
    train_dataset = PlantDataset(str(splits_path / 'train'), transform=get_train_transforms(img_size))
    val_dataset = PlantDataset(str(splits_path / 'val'), transform=get_val_transforms(img_size))
    test_dataset = PlantDataset(str(splits_path / 'test'), transform=get_val_transforms(img_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_dataset.classes, len(train_dataset.classes)
