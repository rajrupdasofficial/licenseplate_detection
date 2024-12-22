import os
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
import matplotlib.pyplot as plt
import logging

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")

# Character Mapping
CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR_TO_INDEX = {char: idx for idx, char in enumerate(CHARS)}
INDEX_TO_CHAR = {idx: char for idx, char in enumerate(CHARS)}

# Advanced Augmentation Transform
def get_augmentation_transform(is_training=True):
    if is_training:
        return A.Compose([
            A.Resize(64, 128),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.3),
                A.ColorJitter(p=0.3)
            ]),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(64, 128),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

# Advanced Dataset Class
class LicensePlateDataset(Dataset):
    def __init__(self, image_dir, is_training=True):
        self.image_dir = image_dir
        self.is_training = is_training

        # Validate image directory
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory not found: {image_dir}")

        # Filter image files
        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if not self.image_files:
            raise ValueError(f"No image files found in {image_dir}")

        # Set up transforms
        self.transform = get_augmentation_transform(is_training)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        transformed = self.transform(image=image)
        image = transformed['image']

        # Process label from filename
        label_text = os.path.splitext(img_name)[0].upper()
        label = [CHAR_TO_INDEX.get(c, 0) for c in label_text[:7]]

        # Pad or truncate label
        label = label + [0] * (7 - len(label))

        return image, torch.tensor(label, dtype=torch.long)

# Advanced Neural Network Model
class AdvancedLicensePlateNet(nn.Module):
    def __init__(self, num_classes=36):
        super().__init__()

        # Feature Extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes * 7)
        )

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features).view(-1, 7, 36)

# Accuracy Calculation
def calculate_accuracy(outputs, labels):
    pred_labels = outputs.argmax(dim=2)
    char_matches = (pred_labels == labels).float()

    char_accuracy = char_matches.mean().item()
    full_matches = char_matches.prod(dim=1)
    full_accuracy = full_matches.mean().item()

    return char_accuracy, full_accuracy

# Training Function
def train_model(
    image_dir,
    epochs=100,
    batch_size=64,
    learning_rate=1e-3,
    model_save_path='licenseplatemodel.pth'
):
    # Dataset and DataLoader
    dataset = LicensePlateDataset(image_dir, is_training=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Model and Optimization
    model = AdvancedLicensePlateNet().to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_char_acc = 0
        total_full_acc = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs.view(-1, 36), labels.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Accuracy Calculation
            char_acc, full_acc = calculate_accuracy(outputs, labels)

            total_loss += loss.item()
            total_char_acc += char_acc
            total_full_acc += full_acc

        # Epoch Summary
        avg_loss = total_loss / len(dataloader)
        avg_char_acc = total_char_acc / len(dataloader)
        avg_full_acc = total_full_acc / len(dataloader)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Character Accuracy: {avg_char_acc:.4f}")
        print(f"Full Plate Accuracy: {avg_full_acc:.4f}\n")

    # Save Model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# Main Execution
if __name__ == "__main__":
    # Update this path to your dataset directory
    IMAGE_DIR = 'datasets/images'

    train_model(IMAGE_DIR)
