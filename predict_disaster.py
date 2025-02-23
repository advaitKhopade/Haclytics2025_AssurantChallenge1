import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Define your model architecture (must match training)
class MultiTaskModel(nn.Module):
    def __init__(self, num_disaster_types):
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

# Initialize the model with the correct number of disaster types (e.g., 8)
model = MultiTaskModel(num_disaster_types=8)

# Load model weights directly from the best_model.pth file
model_state_path = "best_model.pth"
try:
    model.load_state_dict(torch.load(model_state_path, map_location=torch.device("cpu")))
    model.eval()
    print("Model loaded successfully from best_model.pth")
except Exception as e:
    print(f"Error loading model: {e}")

# Example prediction code: loading and transforming an image
image_path = r"C:\Documents\Haclytics2025_AssurantChallenge1\UserInputImagesTest\UserLeak.jpg"
input_image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    disaster_out, severity_out = model(input_tensor)
    disaster_pred = disaster_out.argmax(dim=1).item()
    severity_pred = severity_out.argmax(dim=1).item()
    
print(f"Predicted disaster type index: {disaster_pred}")
print(f"Predicted severity index: {severity_pred}")
