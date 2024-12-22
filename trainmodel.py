import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR_TO_INDEX = {char: idx for idx, char in enumerate(CHARS)}
INDEX_TO_CHAR = {idx: char for idx, char in enumerate(CHARS)}

class LicensePlateDataset(Dataset):
    def __init__(self, image_dir, is_training=True):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.is_training = is_training

        self.train_transform = A.Compose([
            A.Resize(64, 128),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.ISONoise(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
            ], p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        self.val_transform = A.Compose([
            A.Resize(64, 128),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Use OpenCV for image loading
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.is_training:
            image = self.train_transform(image=image)['image']
        else:
            image = self.val_transform(image=image)['image']


        label_text = os.path.splitext(img_name)[0]
        label = np.array([CHAR_TO_INDEX.get(c, 0) for c in label_text.upper()])
        label = np.pad(label, (0, max(0, 7 - len(label))), 'constant')
        return image, torch.tensor(label[:7], dtype=torch.long)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

class EnhancedLicensePlateNet(nn.Module):
    def __init__(self, num_classes=36):
        super(EnhancedLicensePlateNet, self).__init__()


        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self.make_layer(64, 64, 2)
        self.layer2 = self.make_layer(64, 128, 2, stride=2)
        self.layer3 = self.make_layer(128, 256, 2, stride=2)
        self.layer4 = self.make_layer(256, 512, 2, stride=2)


        self.attention = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )


        self.avg_pool = nn.AdaptiveAvgPool2d((1, 2))


        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes * 7)
        )


        self._initialize_weights()

    def make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        att = self.attention(x)
        x = x * att

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x.view(-1, 7, 36)

def train_model(image_dir, epochs=100, batch_size=64, learning_rate=0.002):  # Increased learning rate

    dataset = LicensePlateDataset(image_dir, is_training=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Total images: {len(dataset)}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")


    model = EnhancedLicensePlateNet().to(device)
    criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )


    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=1000.0
    )

    scaler = GradScaler()

    best_accuracy = 0
    patience = 10
    patience_counter = 0

    print(f"Training with learning rate: {learning_rate}")
    print("Starting Training...")

    for epoch in range(epochs):

        model.train()
        train_loss = 0
        train_char_acc = 0
        train_plate_acc = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()


            with autocast():
                outputs = model(images)
                loss = criterion(outputs.reshape(-1, 36), labels.reshape(-1))

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()


            with torch.no_grad():
                _, predicted = torch.max(outputs, 2)
                correct = (predicted == labels)
                char_acc = correct.float().mean().item() * 100
                plate_acc = (correct.sum(dim=1) == 7).float().mean().item() * 100

                train_char_acc += char_acc
                train_plate_acc += plate_acc

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Char Acc: {char_acc:.2f}%, Plate Acc: {plate_acc:.2f}%')


        model.eval()
        val_loss = 0
        val_char_acc = 0
        val_plate_acc = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs.reshape(-1, 36), labels.reshape(-1))
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 2)
                correct = (predicted == labels)
                val_char_acc += correct.float().mean().item() * 100
                val_plate_acc += (correct.sum(dim=1) == 7).float().mean().item() * 100


        train_loss /= len(train_loader)
        train_char_acc /= len(train_loader)
        train_plate_acc /= len(train_loader)
        val_loss /= len(val_loader)
        val_char_acc /= len(val_loader)
        val_plate_acc /= len(val_loader)

        print(f'\nEpoch [{epoch+1}/{epochs}] Summary:')
        print(f'Training - Loss: {train_loss:.4f}, Char Acc: {train_char_acc:.2f}%, Plate Acc: {train_plate_acc:.2f}%')
        print(f'Validation - Loss: {val_loss:.4f}, Char Acc: {val_char_acc:.2f}%, Plate Acc: {val_plate_acc:.2f}%\n')

        if val_char_acc > best_accuracy:
            best_accuracy = val_char_acc
            patience_counter = 0


            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': best_accuracy,
                'char_to_index': CHAR_TO_INDEX,
                'index_to_char': INDEX_TO_CHAR,
                'model_config': {
                    'input_size': (64, 128),
                    'num_classes': 36
                }
            }, 'best_license_plate_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    print("Training completed!")
    print(f"Best validation character accuracy: {best_accuracy:.2f}%")

    model_size = os.path.getsize('trainedmodel.pth') / (1024 * 1024)  # Size in MB
    print(f"Model size: {model_size:.2f} MB")

if __name__ == "__main__":
    IMAGE_DIR = 'datasets/images'

    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Directory {IMAGE_DIR} does not exist!")
    else:
        train_model(IMAGE_DIR, epochs=100, batch_size=64, learning_rate=0.002) #can you please check why the accuracy is not improving even after 30 epoch ? increase everything from learning rate to everything
