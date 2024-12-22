import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import sys

class LicensePlateNet(nn.Module):
    def __init__(self, num_classes=36):
        super(LicensePlateNet, self).__init__()


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
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Dynamic Feature Size Calculation
        self._calculate_feature_size()

    def _calculate_feature_size(self):
        # Dynamic feature size calculation
        test_input = torch.randn(1, 3, 48, 96)
        with torch.no_grad():
            features = self.features(test_input)
            feature_size = features.view(features.size(0), -1).size(1)

        # Advanced Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 36 * 7)
        )

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output.view(-1, 7, 36)

def preprocess_plate_image(plate_img):
    """Enhanced plate image preprocessing"""
    try:

        height, width = plate_img.shape[:2]
        aspect_ratio = width / height
        new_width = 300
        new_height = int(new_width / aspect_ratio)
        plate_resize = cv2.resize(plate_img, (new_width, new_height), interpolation=cv2.INTER_AREA)


        gray = cv2.cvtColor(plate_resize, cv2.COLOR_BGR2GRAY)


        _, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)


        combined_binary = cv2.bitwise_and(binary1, binary2)

        denoised = cv2.fastNlMeansDenoising(combined_binary, None, 10, 7, 21)


        plate_rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)

        return plate_rgb
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return plate_img

def detect_license_plates():

    camera_indices = [0, 1, -1]
    cap = None

    for idx in camera_indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Successfully opened camera at index {idx}")
            break

    if not cap or not cap.isOpened():
        print("❌ No camera found. Connect a camera and retry.")
        return


    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    model = LicensePlateNet().to(device)


    try:
        checkpoint = torch.load('best_license_plate_model.pth',
                                map_location=device,
                                weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

        print("Model loaded successfully!")
    except FileNotFoundError:
        print("No pre-trained model found. Using randomly initialized weights.")
    except Exception as e:
        print(f"Model loading error: {e}")

    model.eval()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 96)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


    CHAR_MAP = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Advanced Plate Detection
            plate_cascades = [
                cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
            ]

            all_plates = []
            for cascade in plate_cascades:
                plates = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(75, 25)
                )
                all_plates.extend(plates)

            for (x, y, w, h) in all_plates:

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


                plate_img = frame[y:y+h, x:x+w]
                preprocessed_plate = preprocess_plate_image(plate_img)


                plate_tensor = transform(preprocessed_plate).unsqueeze(0).to(device)


                with torch.no_grad():
                    output = model(plate_tensor)
                    predicted_chars = torch.argmax(output, dim=2)
                    confidences = torch.max(torch.softmax(output, dim=2), dim=2)[0]


                    license_text = ''.join([
                        CHAR_MAP[idx] if conf > 0.7 else '?'
                        for idx, conf in zip(predicted_chars[0], confidences[0])
                    ])


                    cv2.putText(
                        frame,
                        license_text,
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )


            cv2.imshow('License Plate Detection', frame)
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(f"Error details: {sys.exc_info()}")

    finally:

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_license_plates()
