# data_loader_flip.py
"""
PyTorch data loader for face attribute prediction with horizontal 
flip augmentation, handling age regression, gender classification, 
and eye coordinate detection from facial images.
"""
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from config import (
    BATCH_SIZE,
    DATA_CSV_PATH,
    IMAGE_DIR,
    IMG_HEIGHT,
    IMG_WIDTH,
    MAX_AGE,
    ORIG_IMG_HEIGHT,
    ORIG_IMG_WIDTH,
    RANDOM_STATE,
    VALIDATION_SPLIT,
)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class FaceDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, is_training=False):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.is_training = is_training

        self.image_paths = (
            self.df["im_name"].apply(lambda x: os.path.join(self.image_dir, x)).values
        )

        # Genders, ages y eye coordinates Normalization
        self.genders = (
            self.df["gender"]
            .map({"male": 0.0, "female": 1.0})
            .astype(np.float32)
            .values
        )

        self.ages = (self.df["age"].astype(np.float32) / MAX_AGE).values

        lx_norm = self.df["left_eye_x"].astype(np.float32) / ORIG_IMG_WIDTH
        ly_norm = self.df["left_eye_y"].astype(np.float32) / ORIG_IMG_HEIGHT
        rx_norm = self.df["right_eye_x"].astype(np.float32) / ORIG_IMG_WIDTH
        ry_norm = self.df["right_eye_y"].astype(np.float32) / ORIG_IMG_HEIGHT

        # store individual eye coordinates for flipping
        self.lx_coords = np.clip(lx_norm, 0.0, 1.0).values
        self.ly_coords = np.clip(ly_norm, 0.0, 1.0).values
        self.rx_coords = np.clip(rx_norm, 0.0, 1.0).values
        self.ry_coords = np.clip(ry_norm, 0.0, 1.0).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image not found at {img_path}. Skipping.")

            return None

        lx, ly, rx, ry = (
            self.lx_coords[idx],
            self.ly_coords[idx],
            self.rx_coords[idx],
            self.ry_coords[idx],
        )

        if self.is_training:
            if random.random() < 0.5:
                image = transforms.functional.hflip(image)
                new_lx = 1.0 - rx
                new_rx = 1.0 - lx
                lx, rx = new_lx, new_rx
            if self.transform:
                image = self.transform(image)
        else:
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)

        age = torch.tensor(self.ages[idx], dtype=torch.float32)
        gender = torch.tensor(self.genders[idx], dtype=torch.float32).unsqueeze(0)
        eyes = torch.tensor([lx, ly, rx, ry], dtype=torch.float32)

        return image, (age, gender, eyes)


def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None, (None, None, None)
    return torch.utils.data.dataloader.default_collate(batch)


def get_data_loaders():
    df_full = pd.read_csv(DATA_CSV_PATH)
    df_full["image_path_check"] = df_full["im_name"].apply(
        lambda x: os.path.join(IMAGE_DIR, x)
    )
    df_full = df_full[df_full["image_path_check"].apply(os.path.exists)].drop(
        columns=["image_path_check"]
    )

    train_df, val_df = train_test_split(
        df_full, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE
    )

    # Transformations
    # For MobileNetV2, ImageNet normalization stats
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = FaceDataset(
        train_df, IMAGE_DIR, transform=train_transform, is_training=True
    )
    val_dataset = FaceDataset(
        val_df, IMAGE_DIR, transform=val_transform, is_training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn_skip_none,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn_skip_none,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    for images, (ages, genders, eyes) in train_loader:
        if images is None:
            print("Skipped a batch due to missing images.")
            continue
        print("Images shape:", images.shape)
        print("Ages shape:", ages.shape)
        print("Genders shape:", genders.shape)
        print("Eyes shape:", eyes.shape)
        print("Sample age:", ages[0].item())
        print("Sample gender:", genders[0].item())
        print("Sample eyes (potentialy flipped and adjusted):", eyes[0].np())
        break
