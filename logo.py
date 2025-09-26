import ssl
print(ssl.get_default_verify_paths())
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
import torchvision.models as models
import os
from torchvision.models import ResNet18_Weights
from ultralytics import YOLO

from ultralytics import YOLO
import cv2

logo_model = YOLO("best.pt")

classifier = models.resnet18(weights=ResNet18_Weights.DEFAULT)
classifier.fc = nn.Sequential(
    nn.Linear(classifier.fc.in_features, 1),
    nn.Sigmoid()
)
classifier.load_state_dict(torch.load("real_fake_logo_classifier.pth", map_location="cpu"))
classifier.eval()

# Preprocessing for classifier
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def detect_logo_and_crop(img_path, conf=0.5, transform=transform):
    results = logo_model.predict(source=img_path, conf=conf, save=False)
    pred_bboxes = results[0].boxes.xyxy.cpu().numpy()

    if len(pred_bboxes) == 0:
        return None  # No logo detected

    # Take first detected logo
    xmin, ymin, xmax, ymax = map(int, pred_bboxes[0])
    img = cv2.imread(img_path)
    cropped_logo = img[ymin:ymax, xmin:xmax]

    return cropped_logo, (xmin, ymin, xmax, ymax)

# Load classifier
num_classes = ['Real', 'Fake']

def predict(image_path):
    result = detect_logo_and_crop(image_path)

    img_cv = cv2.imread(image_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    if result is None:
        return img_cv, []  # No logo detected

    cropped_logo, bbox = result

    # Convert cropped logo to PIL
    cropped_pil = Image.fromarray(cv2.cvtColor(cropped_logo, cv2.COLOR_BGR2RGB))
    image_tensor = transform(cropped_pil).unsqueeze(0)

    with torch.no_grad():
        output = classifier(image_tensor)
        predicted_label = "Real" if output.item() > 0.5 else "Fake"

    # Draw bounding box on original image
    img_cv = cv2.imread(image_path)
    x1, y1, x2, y2 = bbox
    color = (0, 255, 0) if predicted_label == "Real" else (0, 0, 255)
    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img_cv, predicted_label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return img_cv, [predicted_label]