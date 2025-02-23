#!/usr/bin/env python3

import os
import random
import json  # <-- Added for saving metadata as JSON
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# Function to analyze per-class performance (if needed elsewhere)
def analyze_class_performance(labels_true, labels_pred, class_names):
    """Analyze per-class performance metrics"""
    performance = {}
    for i, class_name in enumerate(class_names):
        mask = labels_true == i
        if np.sum(mask) > 0:
            accuracy = np.mean(labels_pred[mask] == labels_true[mask])
            performance[class_name] = {
                'accuracy': accuracy,
                'samples': np.sum(mask)
            }
    return performance

# Dataset class definition
class DisasterDataset(Dataset):
    def __init__(self, data_df, img_dir, transform=None):
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

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Image Name'])
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        disaster_label = self.disaster2idx[row['Disaster Type']]
        severity_label = int(row['Severity'])
        
        return {
            'image': image,
            'disaster_type': disaster_label,
            'severity': severity_label,
            'weight': self.disaster_weights[disaster_label]
        }

# Model definition with ResNet50 backbone
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

# Training function for one epoch
def train_epoch(model, train_loader, optimizer, device, criterion):
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
        
        # Calculate weighted disaster loss - ensure it's a scalar
        disaster_loss = criterion(disaster_out, disaster_labels)
        disaster_loss = (disaster_loss * weights).mean()
        
        # Calculate severity loss - ensure it's a scalar
        severity_loss = criterion(severity_out, severity_labels).mean()
        
        # Combined loss
        loss = disaster_loss + severity_loss
        loss.backward()
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

# Evaluation function
def evaluate(model, data_loader, device, criterion):
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

# Main function that implements data splits, training, and analysis
def main():
    # Configuration
    RANDOM_SEED = 42
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    
    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # Data paths
    img_dir = "images"
    csv_path = "disasterData/disaster_analysis_filename_based.csv"
    
    # Load data from CSV
    df = pd.read_csv(csv_path)
    
    # Analyze class distribution and compute significance scores
    class_counts = df['Disaster Type'].value_counts()
    total_samples = len(df)
    significance_scores = {}
    min_class_ratio = 0.02  # Minimum 2% of total samples
    min_absolute_samples = 5  # Minimum number of samples per class
    
    print("\nClass Distribution Analysis:")
    for disaster_type, count in class_counts.items():
        ratio = count / total_samples
        significance_score = ratio * (count / min_absolute_samples)
        significance_scores[disaster_type] = significance_score
        
        print(f"{disaster_type}:")
        print(f"  Samples: {count}")
        print(f"  Ratio: {ratio:.3f}")
        print(f"  Significance Score: {significance_score:.3f}")
    
    # Select significant classes based on threshold
    significance_threshold = 0.1
    valid_classes = [
        cls for cls, score in significance_scores.items()
        if score >= significance_threshold and class_counts[cls] >= min_absolute_samples
    ]
    
    # Create custom train/test splits ensuring minimum samples per class
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    min_test_size = 2  # At least 2 samples per class for test set
    
    for disaster_type in valid_classes:
        class_data = df[df['Disaster Type'] == disaster_type]
        n_samples = len(class_data)
        n_test = max(min_test_size, int(n_samples * 0.3))  # Use 30% for test
        class_test = class_data.sample(n=n_test, random_state=RANDOM_SEED)
        class_train = class_data.drop(class_test.index)
        train_df = pd.concat([train_df, class_train])
        test_df = pd.concat([test_df, class_test])
    
    # Further split test set into validation and test
    val_df, test_df = train_test_split(
        test_df, 
        test_size=0.5, 
        random_state=RANDOM_SEED,
        stratify=test_df['Disaster Type']
    )
    
    # Shuffle datasets
    train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    print(f"\nSelected {len(valid_classes)} significant classes")
    print("Selected classes:", valid_classes)
    print("\nFinal dataset sizes:")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Data transforms for training and evaluation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets and dataloaders
    train_dataset = DisasterDataset(train_df, img_dir, transform=train_transform)
    val_dataset = DisasterDataset(val_df, img_dir, transform=eval_transform)
    test_dataset = DisasterDataset(test_df, img_dir, transform=eval_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Setup model and training components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel(num_disaster_types=len(train_dataset.disaster_types))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop with model checkpointing
    best_val_acc = 0
    for epoch in range(NUM_EPOCHS):
        train_metrics = train_epoch(model, train_loader, optimizer, device, criterion)
        val_metrics = evaluate(model, val_loader, device, criterion)
        
        # Save best model based on disaster accuracy
        if val_metrics['disaster_acc'] > best_val_acc:
            best_val_acc = val_metrics['disaster_acc']
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Train Disaster Acc: {train_metrics['disaster_acc']:.4f}")
        print(f"Train Severity Acc: {train_metrics['severity_acc']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Disaster Acc: {val_metrics['disaster_acc']:.4f}")
        print(f"Val Severity Acc: {val_metrics['severity_acc']:.4f}")
    
    # Load the best saved model and evaluate on the test set
    model.load_state_dict(torch.load('best_model.pth'))
    test_metrics = evaluate(model, test_loader, device, criterion)
    
    print("\nTest Results:")
    print(f"Test Disaster Accuracy: {test_metrics['disaster_acc']:.4f}")
    print(f"Test Severity Accuracy: {test_metrics['severity_acc']:.4f}")
    
    # Generate and display classification reports
    disaster_report = classification_report(
        test_metrics['labels']['disaster'],
        test_metrics['predictions']['disaster'],
        target_names=train_dataset.disaster_types,
        zero_division=0
    )
    
    severity_report = classification_report(
        test_metrics['labels']['severity'],
        test_metrics['predictions']['severity'],
        target_names=['Low', 'Medium', 'High'],
        zero_division=0
    )
    
    print("\nDisaster Type Classification Report:")
    print(disaster_report)
    
    print("\nSeverity Classification Report:")
    print(severity_report)
    
    # Analyze per-class performance and identify problematic classes
    print("\nFeature Importance Analysis:")
    class_performance = {}
    for i, disaster_type in enumerate(train_dataset.disaster_types):
        mask = test_metrics['labels']['disaster'] == i
        if np.sum(mask) > 0:
            accuracy = np.mean(
                test_metrics['predictions']['disaster'][mask] == 
                test_metrics['labels']['disaster'][mask]
            )
            class_performance[disaster_type] = accuracy
            print(f"{disaster_type}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Samples: {np.sum(mask)}")
    
    problem_threshold = 0.7  # Adjust threshold as needed
    problematic_classes = [
        cls for cls, acc in class_performance.items()
        if acc < problem_threshold
    ]
    
    if problematic_classes:
        print("\nProblematic classes that might need more data or feature engineering:")
        for cls in problematic_classes:
            print(f"- {cls} (Accuracy: {class_performance[cls]:.4f})")
    
    # Save the disaster types mapping for later use (e.g., in prediction)
    print("\nSaving disaster types to file...")
    with open('disaster_types.txt', 'w') as f:
        for disaster_type in train_dataset.disaster_types:
            f.write(f"{disaster_type}\n")
    print("Disaster types saved successfully")
    
    # --- Save model metadata to a JSON file ---
    metadata = {
        "model_architecture": "MultiTaskModel",
        "backbone": "ResNet50",
        "num_disaster_types": len(train_dataset.disaster_types),
        "training_hyperparameters": {
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "random_seed": RANDOM_SEED,
            "image_size": IMG_SIZE
        },
        "performance": {
            "best_validation_disaster_accuracy": best_val_acc,
            "test_disaster_accuracy": test_metrics['disaster_acc'],
            "test_severity_accuracy": test_metrics['severity_acc']
        },
        "disaster_types": train_dataset.disaster_types,
        "model_state_file": "best_model.pth"
    }
    
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print("Model metadata saved successfully to model_metadata.json")

if __name__ == "__main__":
    main()
