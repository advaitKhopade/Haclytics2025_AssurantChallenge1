import os
import json
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import classification_report
from PIL import Image

# Configure logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DisasterDataset(Dataset):
    """Dataset class for disaster images with multi-task labels."""
    
    def __init__(self, data_df: pd.DataFrame, img_dir: str, transform: Optional[transforms.Compose] = None):
        """
        Initialize the dataset.
        
        Args:
            data_df: DataFrame containing image paths and labels
            img_dir: Directory containing the images
            transform: Optional transforms to be applied to images
        """
        self.data_df = data_df.copy()
        self.img_dir = img_dir
        self.transform = transform
        
        # Create label mappings
        self.disaster_types = sorted(data_df['Disaster Type'].unique())
        self.disaster2idx = {disaster: idx for idx, disaster in enumerate(self.disaster_types)}
        
        # Calculate class weights for disaster types
        disaster_counts = data_df['Disaster Type'].value_counts()
        total_samples = len(data_df)
        self.disaster_weights = torch.FloatTensor([
            total_samples / (len(disaster_counts) * count) 
            for disaster in self.disaster_types
            for count in [disaster_counts[disaster]]
        ])

    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data_df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Image Name'])
        
        try:
            # Load and transform image
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            # Get labels
            disaster_label = self.disaster2idx[row['Disaster Type']]
            severity_label = int(row['Severity'])
            
            return {
                'image': image,
                'disaster_type': torch.tensor(disaster_label, dtype=torch.long),
                'severity': torch.tensor(severity_label, dtype=torch.long),
                'weight': self.disaster_weights[disaster_label]
            }
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            raise

class MultiTaskModel(nn.Module):
    """Multi-task model for disaster classification and severity prediction."""
    
    def __init__(self, num_disaster_types: int, dropout_rate: float = 0.6):
        """
        Initialize the model.
        
        Args:
            num_disaster_types: Number of disaster classes
            dropout_rate: Dropout rate for regularization
        """
        super(MultiTaskModel, self).__init__()
        
        # Use ResNet34 backbone
        self.backbone = models.resnet34(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Task-specific heads with increased dropout
        self.disaster_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_disaster_types)
        )
        
        self.severity_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 3)  # 3 severity levels
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x).squeeze(-1).squeeze(-1)
        disaster_out = self.disaster_classifier(features)
        severity_out = self.severity_classifier(features)
        return disaster_out, severity_out

class EarlyStopping:
    """Early stopping handler to prevent overfitting."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    clip_grad_norm: float = 1.0
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    disaster_correct = 0
    severity_correct = 0
    total_samples = 0
    
    for batch in train_loader:
        images = batch['image'].to(device)
        disaster_labels = batch['disaster_type'].to(device)
        severity_labels = batch['severity'].to(device)
        weights = batch['weight'].to(device)
        
        optimizer.zero_grad()
        disaster_out, severity_out = model(images)
        
        # Calculate weighted losses
        disaster_loss = criterion(disaster_out, disaster_labels)
        disaster_loss = (disaster_loss * weights).mean()
        severity_loss = criterion(severity_out, severity_labels).mean()
        loss = disaster_loss + severity_loss
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        
        # Track metrics
        with torch.no_grad():
            disaster_pred = disaster_out.argmax(dim=1)
            severity_pred = severity_out.argmax(dim=1)
            disaster_correct += (disaster_pred == disaster_labels).sum().item()
            severity_correct += (severity_pred == severity_labels).sum().item()
            total_samples += images.size(0)
            total_loss += loss.item()
    
    return {
        'loss': total_loss / len(train_loader),
        'disaster_acc': disaster_correct / total_samples,
        'severity_acc': severity_correct / total_samples
    }

def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    disaster_correct = 0
    severity_correct = 0
    total_samples = 0
    
    all_disaster_preds = []
    all_severity_preds = []
    all_disaster_labels = []
    all_severity_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            disaster_labels = batch['disaster_type'].to(device)
            severity_labels = batch['severity'].to(device)
            
            disaster_out, severity_out = model(images)
            
            # Calculate losses
            disaster_loss = criterion(disaster_out, disaster_labels).mean()
            severity_loss = criterion(severity_out, severity_labels).mean()
            loss = disaster_loss + severity_loss
            
            # Track predictions
            disaster_pred = disaster_out.argmax(dim=1)
            severity_pred = severity_out.argmax(dim=1)
            
            disaster_correct += (disaster_pred == disaster_labels).sum().item()
            severity_correct += (severity_pred == severity_labels).sum().item()
            total_samples += images.size(0)
            total_loss += loss.item()
            
            all_disaster_preds.extend(disaster_pred.cpu().numpy())
            all_severity_preds.extend(severity_pred.cpu().numpy())
            all_disaster_labels.extend(disaster_labels.cpu().numpy())
            all_severity_labels.extend(severity_labels.cpu().numpy())
    
    return {
        'loss': total_loss / len(data_loader),
        'disaster_acc': disaster_correct / total_samples,
        'severity_acc': severity_correct / total_samples,
        'predictions': {
            'disaster': np.array(all_disaster_preds),
            'severity': np.array(all_severity_preds)
        },
        'labels': {
            'disaster': np.array(all_disaster_labels),
            'severity': np.array(all_severity_labels)
        }
    }

def save_model(
    model: nn.Module,
    disaster_types: list,
    save_dir: str = 'model_artifacts'
) -> Tuple[str, str]:
    """Save model and metadata."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model state
    model_path = os.path.join(save_dir, f'model_{timestamp}.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save scripted model for production
    scripted_model = torch.jit.script(model)
    script_path = os.path.join(save_dir, f'model_{timestamp}_scripted.pt')
    scripted_model.save(script_path)
    
    # Save metadata
    metadata = {
        'disaster_types': disaster_types,
        'timestamp': timestamp,
        'model_path': model_path,
        'script_path': script_path
    }
    
    metadata_path = os.path.join(save_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model saved to {model_path}")
    return model_path, metadata_path

def load_model(
    model_path: str,
    num_disaster_types: int,
    device: torch.device
) -> nn.Module:
    """Load a saved model."""
    model = MultiTaskModel(num_disaster_types=num_disaster_types)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def get_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get training and evaluation transforms."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, eval_transform

def predict_single_image(
    model: nn.Module,
    image_path: str,
    transform: transforms.Compose,
    device: torch.device,
    disaster_types: list
) -> Dict[str, str]:
    """Predict disaster type and severity for a single image."""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            disaster_out, severity_out = model(image_tensor)
            disaster_pred = disaster_out.argmax(dim=1).item()
            severity_pred = severity_out.argmax(dim=1).item()
        
        return {
            'disaster_type': disaster_types[disaster_pred],
            'severity': ['Low', 'Medium', 'High'][severity_pred]
        }
    except Exception as e:
        logger.error(f"Error predicting image {image_path}: {str(e)}")
        raise