import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import sys
import pygame
import torch.nn.functional as F
import pytesseract

# Advanced License Plate Detection Model
class EnhancedLicensePlateNet(nn.Module):
    def __init__(self, num_classes=36):
        super(EnhancedLicensePlateNet, self).__init__()

        # Enhanced Feature Extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Advanced Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes * 7)  # 7 characters, 36 possible classes
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x.view(-1, 7, 36)

# Character Mapping
CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR_TO_INDEX = {char: idx for idx, char in enumerate(CHARS)}
INDEX_TO_CHAR = {idx: char for idx, char in enumerate(CHARS)}

def setup_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Camera not accessible!")
        return None
    return cap

def load_model(device):
    try:
        model = EnhancedLicensePlateNet().to(device)

        try:
            # Try loading pre-trained weights
            checkpoint = torch.load('licenseplatemodel.pth', map_location=device)

            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']

            model.load_state_dict(checkpoint, strict=False)
            print("Model loaded successfully!")
        except Exception as load_error:
            print(f"Model loading error: {load_error}")
            print("Using randomly initialized model for testing")

        model.eval()
        return model

    except Exception as e:
        print(f"Catastrophic model loading failure: {e}")
        sys.exit(1)

def detect_license_plates(model, frame, transform, device):
    try:
        # Preprocessing
        frame_resized = cv2.resize(frame, (224, 224))
        input_tensor = transform(frame_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=2)

            # Get top predictions for each character position
            top_predictions = []
            for pos in range(7):
                char_probs = probabilities[0, pos, :]
                top_val, top_idx = torch.topk(char_probs, k=1)

                if top_val.item() > 0.5:  # Confidence threshold
                    top_predictions.append((pos, CHARS[top_idx.item()], top_val.item()))

        # Construct license plate text
        license_text = ''.join([char for _, char, _ in sorted(top_predictions)])

        return license_text

    except Exception as e:
        print(f"Detection Error: {e}")
        return ""

def detect_license_plates_with_ocr(frame):
    # Preprocessing for OCR
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Apply noise reduction
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)

    # Pytesseract OCR
    try:
        license_text = pytesseract.image_to_string(
            denoised,
            config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ).strip()

        # Basic validation
        if len(license_text) >= 6 and len(license_text) <= 10:
            return license_text
    except Exception as e:
        print(f"OCR Error: {e}")

    return ""

def detect_license_plates_pygame():
    cap = setup_camera()
    if not cap:
        print("Camera setup failed. Exiting.")
        return

    pygame.init()
    screen_width, screen_height = 1280, 720
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Advanced License Plate Detection')
    clock = pygame.time.Clock()

    # Font for text overlay
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 50)

    # Device and Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)

    # Transform for neural network
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Detection variables
    license_text = ""
    detection_timer = 0
    detection_methods = ['neural', 'ocr']
    current_method = 0

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    # Toggle detection method
                    current_method = (current_method + 1) % len(detection_methods)

            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Periodic Detection
            detection_timer += 1
            if detection_timer >= 30:
                if detection_methods[current_method] == 'neural':
                    # Neural Network Detection
                    license_text = detect_license_plates(model, frame, transform, device)
                else:
                    # OCR Detection
                    license_text = detect_license_plates_with_ocr(frame)

                detection_timer = 0

            # Convert frame to RGB for Pygame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))

            screen.blit(frame_surface, (0, 0))

            # Render License Plate Text with Method Indication
            if license_text:
                text_surface = font.render(
                    f"License: {license_text} [{detection_methods[current_method]}]",
                    True,
                    (0, 255, 0)
                )
                screen.blit(text_surface, (50, 50))

            pygame.display.flip()
            clock.tick(30)

    except Exception as e:
        print(f"Pygame Detection Failure: {e}")
        import traceback
        traceback.print_exc()

    finally:
        cap.release()
        pygame.quit()

if __name__ == "__main__":
    detect_license_plates_pygame()
