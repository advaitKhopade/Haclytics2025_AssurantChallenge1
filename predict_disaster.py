# predict_disaster.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

class MultiTaskModel(nn.Module):
    def __init__(self, num_disaster_types=8):
        super(MultiTaskModel, self).__init__()
        # Use ResNet50 backbone
        self.backbone = models.resnet50(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Task-specific heads with dropout
        self.disaster_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_disaster_types)
        )
        self.severity_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 3)
        )

    def forward(self, x):
        features = self.backbone(x).squeeze(-1).squeeze(-1)
        disaster_out = self.disaster_classifier(features)
        severity_out = self.severity_classifier(features)
        return disaster_out, severity_out


def load_model(model_path="best_model.pth", num_disaster_types=8):
    model = MultiTaskModel(num_disaster_types=num_disaster_types)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def predict_disaster(image_path, model=None):
    """Predict the disaster type and severity for a single image."""

    # If no model provided, load it fresh (not optimal for multiple requests)
    if model is None:
        model = load_model()

    # Transforms (must match training normalization, etc.)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load and preprocess the image
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)  # add batch dimension

    # Inference
    with torch.no_grad():
        disaster_out, severity_out = model(input_tensor)
        disaster_idx = disaster_out.argmax(dim=1).item()
        severity_idx = severity_out.argmax(dim=1).item()

    # If you have a separate `disaster_types.txt`, load from there:
    # For simplicity, we hardcode them here:
    disaster_types = [
        "earthquake", "fire", "flood", "leak",
        "storm", "structural damage", "surface damage", "tornado"
    ]
    severity_levels = ["Low", "Medium", "High"]

    return {
        "disaster_type": disaster_types[disaster_idx],
        "severity": severity_levels[severity_idx]
    }
