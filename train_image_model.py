"""
Fish Species & Freshness Classification Model Training
CNN with Transfer Learning (ResNet50/EfficientNet)
Handles multiple datasets with different structures
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FishDataset(Dataset):
    """Unified Fish Dataset supporting multiple folder structures"""
    
    def __init__(self, image_paths, species_labels, freshness_labels, transform=None):
        self.image_paths = image_paths
        self.species_labels = species_labels
        self.freshness_labels = freshness_labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            species_label = self.species_labels[idx]
            freshness_label = self.freshness_labels[idx]
            
            return image, species_label, freshness_label
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a black image as fallback
            if self.transform:
                dummy_image = self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
            else:
                dummy_image = Image.new('RGB', (224, 224), (0, 0, 0))
            return dummy_image, self.species_labels[idx], self.freshness_labels[idx]

class DatasetUnifier:
    """Unifies all fish datasets into consistent format"""
    
    def __init__(self):
        # Species name mapping for consistency
        self.species_mapping = {
            'Indian Mackerel (Ooli) - Goatfish (Oola)': 'Indian Mackerel',
            'Indian Mackerel(Oozi)-Goatfish(Oola)': 'Indian Mackerel',
            'Indian Anchovy (Nethili)': 'Anchovy', 
            'Indian Anchovy(Nethili)': 'Anchovy',
            'Indian Oil Sardine (Maththi) - Sardine (Saala)': 'Oil Sardine',
            'Indian Oil Sardine(Maththi)-Sardine(Saala)': 'Oil Sardine',
            'Seer Fish-King Mackerel (Nagara)': 'Seer Fish',
            'Seer Fish -King Mackerel(Nagara)': 'Seer Fish',
            'Red Seer Fish (Sevapu Nagara)': 'Seer Fish',
            'Red Seer Fish(Sevapu Nagara)': 'Seer Fish',
            'Ribbon Fish (Kanavai)': 'Ribbon Fish',
            'Ribbon Fish(Kanavai)': 'Ribbon Fish',
            'Barramundi - Sea Bass (Vaaval)': 'Barramundi',
            'Barramundi -Sea Bass(Vaaval)': 'Barramundi',
            'Catfish-(Keluthi)-(Kelangan)': 'Catfish',
            'Catfish': 'Catfish',
            'Silver Catfish': 'Catfish',
            'Trevally-Rockfish (Paara Meen)': 'Trevally',
            'Trevally-Rockfish(Paara Meen)': 'Trevally',
            'Trevally (Velaa meen)': 'Trevally',
            'Trevally(Velaa Meen)': 'Trevally',
            'White Grouper (Vellai Kelangan)': 'White Grouper',
            'White Grouper(Vellai kelangan)': 'White Grouper',
            'Snakehead Fish (Murrel Meen) - Longnose Garfish (Mooku Oola)': 'Snakehead',
            'Snakehead Fish(Murrel Meen)-Longnose Goatfish(Mooku Oola)': 'Snakehead',
            'Jelabi Kenda fish': 'Jelabi Kenda',
            'Jelabi Kenda': 'Jelabi Kenda',
            'Mackerel': 'Indian Mackerel',
            'Black Pomfret': 'Pomfret',
            'Pomfret': 'Pomfret',
            'Indian Carp': 'Indian Carp',
            'Pink Perch': 'Pink Perch',
            'Black Snapper': 'Black Snapper',
            'Prawn': 'Prawn'
        }
        
        self.species_encoder = LabelEncoder()
        self.freshness_encoder = LabelEncoder()
        
    def process_fish_market_dataset(self, base_path):
        """Process E:/sea/datasets/fish_market"""
        print("Processing fish_market dataset...")
        image_paths, species_labels, freshness_labels = [], [], []
        
        for species_folder in os.listdir(base_path):
            species_path = os.path.join(base_path, species_folder)
            if not os.path.isdir(species_path):
                continue
                
            # Map to standardized species name
            species_name = self.species_mapping.get(species_folder, species_folder)
            
            for image_file in os.listdir(species_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(species_path, image_file)
                    image_paths.append(image_path)
                    species_labels.append(species_name)
                    freshness_labels.append('Unknown')  # Default for species-only dataset
        
        print(f"  Found {len(image_paths)} images from {len(set(species_labels))} species")
        return image_paths, species_labels, freshness_labels
    
    def process_freshness_dataset(self, base_path):
        """Process E:/sea/datasets/fresh and non-fresh fish"""
        print("Processing fresh and non-fresh fish dataset...")
        image_paths, species_labels, freshness_labels = [], [], []
        
        for freshness_folder in ['Fresh', 'Not Fresh']:
            freshness_path = os.path.join(base_path, freshness_folder)
            if not os.path.exists(freshness_path):
                continue
                
            freshness_label = 'Fresh' if freshness_folder == 'Fresh' else 'Non-Fresh'
            
            for image_file in os.listdir(freshness_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(freshness_path, image_file)
                    image_paths.append(image_path)
                    species_labels.append('Unknown')  # Species not specified
                    freshness_labels.append(freshness_label)
        
        print(f"  Found {len(image_paths)} images with freshness labels")
        return image_paths, species_labels, freshness_labels
    
    def process_fish_dataset(self, base_path):
        """Process E:/sea/datasets/Fish dataset"""
        print("Processing Fish dataset...")
        image_paths, species_labels, freshness_labels = [], [], []
        
        for species_folder in os.listdir(base_path):
            species_path = os.path.join(base_path, species_folder)
            if not os.path.isdir(species_path):
                continue
                
            species_name = self.species_mapping.get(species_folder, species_folder)
            
            for image_file in os.listdir(species_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(species_path, image_file)
                    image_paths.append(image_path)
                    species_labels.append(species_name)
                    freshness_labels.append('Unknown')
        
        print(f"  Found {len(image_paths)} images from {len(set(species_labels))} species")
        return image_paths, species_labels, freshness_labels
    
    def process_rohu_dataset(self, base_path):
        """Process E:/sea/datasets/Rohu (freshness-specific)"""
        print("Processing Rohu dataset...")
        image_paths, species_labels, freshness_labels = [], [], []
        
        for split in ['Training', 'Testing']:
            split_path = os.path.join(base_path, split)
            if not os.path.exists(split_path):
                continue
                
            for freshness_type in ['Fresh_Eyes', 'Fresh_Gills', 'Nonfresh_Eyes', 'Nonfresh_Gills']:
                type_path = os.path.join(split_path, freshness_type)
                if not os.path.exists(type_path):
                    continue
                    
                freshness_label = 'Fresh' if 'Fresh' in freshness_type else 'Non-Fresh'
                
                for image_file in os.listdir(type_path):
                    if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        image_path = os.path.join(type_path, image_file)
                        image_paths.append(image_path)
                        species_labels.append('Rohu')
                        freshness_labels.append(freshness_label)
        
        print(f"  Found {len(image_paths)} Rohu images with freshness labels")
        return image_paths, species_labels, freshness_labels
    
    def unify_datasets(self, dataset_paths):
        """Combine all datasets into unified format"""
        print("🔄 Unifying all datasets...")
        
        all_image_paths = []
        all_species_labels = []
        all_freshness_labels = []
        
        # Process each dataset
        if 'fish_market' in dataset_paths:
            paths, species, fresh = self.process_fish_market_dataset(dataset_paths['fish_market'])
            all_image_paths.extend(paths)
            all_species_labels.extend(species)
            all_freshness_labels.extend(fresh)
        
        if 'freshness' in dataset_paths:
            paths, species, fresh = self.process_freshness_dataset(dataset_paths['freshness'])
            all_image_paths.extend(paths)
            all_species_labels.extend(species)
            all_freshness_labels.extend(fresh)
        
        if 'fish_dataset' in dataset_paths:
            paths, species, fresh = self.process_fish_dataset(dataset_paths['fish_dataset'])
            all_image_paths.extend(paths)
            all_species_labels.extend(species)
            all_freshness_labels.extend(fresh)
        
        if 'rohu' in dataset_paths:
            paths, species, fresh = self.process_rohu_dataset(dataset_paths['rohu'])
            all_image_paths.extend(paths)
            all_species_labels.extend(species)
            all_freshness_labels.extend(fresh)
        
        # Clean and encode labels
        all_species_labels = [s if s != 'Unknown' else 'Mixed_Species' for s in all_species_labels]
        all_freshness_labels = [f if f != 'Unknown' else 'Fresh' for f in all_freshness_labels]  # Default assumption
        
        # Fit encoders
        self.species_encoder.fit(all_species_labels)
        self.freshness_encoder.fit(all_freshness_labels)
        
        # Encode labels
        species_encoded = self.species_encoder.transform(all_species_labels)
        freshness_encoded = self.freshness_encoder.transform(all_freshness_labels)
        
        print(f"\n✅ Dataset Summary:")
        print(f"  Total images: {len(all_image_paths):,}")
        print(f"  Species classes: {len(self.species_encoder.classes_)}")
        print(f"  Species: {list(self.species_encoder.classes_)}")
        print(f"  Freshness classes: {len(self.freshness_encoder.classes_)}")
        print(f"  Freshness: {list(self.freshness_encoder.classes_)}")
        
        return all_image_paths, species_encoded, freshness_encoded

class FishCNN(nn.Module):
    """Multi-task CNN for species and freshness classification"""
    
    def __init__(self, num_species_classes, num_freshness_classes, model_name='resnet50'):
        super(FishCNN, self).__init__()
        
        # Load pretrained backbone
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final layer
        elif model_name == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=True)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        
        # Multi-task heads
        self.species_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_species_classes)
        )
        
        self.freshness_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_freshness_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        species_out = self.species_classifier(features)
        freshness_out = self.freshness_classifier(features)
        return species_out, freshness_out

class FishModelTrainer:
    """Complete training pipeline for fish classification"""
    
    def __init__(self, model_name='resnet50', batch_size=32, learning_rate=0.001):
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def create_data_loaders(self, image_paths, species_labels, freshness_labels, train_split=0.8):
        """Create training and validation data loaders"""
        
        # Split data
        n_samples = len(image_paths)
        indices = np.random.permutation(n_samples)
        train_size = int(train_split * n_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create datasets
        train_paths = [image_paths[i] for i in train_indices]
        train_species = [species_labels[i] for i in train_indices]
        train_freshness = [freshness_labels[i] for i in train_indices]
        
        val_paths = [image_paths[i] for i in val_indices]
        val_species = [species_labels[i] for i in val_indices]
        val_freshness = [freshness_labels[i] for i in val_indices]
        
        train_dataset = FishDataset(train_paths, train_species, train_freshness, self.train_transform)
        val_dataset = FishDataset(val_paths, val_species, val_freshness, self.val_transform)
        
        # Create weighted sampler for balanced training
        species_counts = Counter(train_species)
        species_weights = {cls: 1.0 / count for cls, count in species_counts.items()}
        sample_weights = [species_weights[species] for species in train_species]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        print(f"Training samples: {len(train_dataset):,}")
        print(f"Validation samples: {len(val_dataset):,}")
        
        return train_loader, val_loader
    
    def train_model(self, train_loader, val_loader, num_species_classes, num_freshness_classes, epochs=50):
        """Train the multi-task CNN model"""
        
        # Initialize model
        model = FishCNN(num_species_classes, num_freshness_classes, self.model_name)
        model = model.to(self.device)
        
        # Loss functions and optimizer
        species_criterion = nn.CrossEntropyLoss()
        freshness_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)
        
        # Training history
        history = {
            'train_species_loss': [], 'train_freshness_loss': [], 'train_total_loss': [],
            'val_species_loss': [], 'val_freshness_loss': [], 'val_total_loss': [],
            'val_species_acc': [], 'val_freshness_acc': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 40)
            
            # Training phase
            model.train()
            train_species_loss = 0.0
            train_freshness_loss = 0.0
            
            for batch_idx, (images, species_targets, freshness_targets) in enumerate(train_loader):
                images = images.to(self.device)
                species_targets = species_targets.to(self.device)
                freshness_targets = freshness_targets.to(self.device)
                
                optimizer.zero_grad()
                
                species_outputs, freshness_outputs = model(images)
                
                species_loss = species_criterion(species_outputs, species_targets)
                freshness_loss = freshness_criterion(freshness_outputs, freshness_targets)
                total_loss = species_loss + 0.5 * freshness_loss  # Weight freshness less
                
                total_loss.backward()
                optimizer.step()
                
                train_species_loss += species_loss.item()
                train_freshness_loss += freshness_loss.item()
                
                if batch_idx % 50 == 0:
                    print(f"  Batch {batch_idx:3d}: Species Loss: {species_loss.item():.4f}, "
                          f"Freshness Loss: {freshness_loss.item():.4f}")
            
            # Validation phase
            model.eval()
            val_species_loss = 0.0
            val_freshness_loss = 0.0
            species_correct = 0
            freshness_correct = 0
            total_samples = 0
            
            with torch.no_grad():
                for images, species_targets, freshness_targets in val_loader:
                    images = images.to(self.device)
                    species_targets = species_targets.to(self.device)
                    freshness_targets = freshness_targets.to(self.device)
                    
                    species_outputs, freshness_outputs = model(images)
                    
                    species_loss = species_criterion(species_outputs, species_targets)
                    freshness_loss = freshness_criterion(freshness_outputs, freshness_targets)
                    
                    val_species_loss += species_loss.item()
                    val_freshness_loss += freshness_loss.item()
                    
                    # Calculate accuracy
                    _, species_predicted = torch.max(species_outputs.data, 1)
                    _, freshness_predicted = torch.max(freshness_outputs.data, 1)
                    
                    total_samples += species_targets.size(0)
                    species_correct += (species_predicted == species_targets).sum().item()
                    freshness_correct += (freshness_predicted == freshness_targets).sum().item()
            
            # Calculate averages
            train_species_loss /= len(train_loader)
            train_freshness_loss /= len(train_loader)
            val_species_loss /= len(val_loader)
            val_freshness_loss /= len(val_loader)
            
            species_acc = 100. * species_correct / total_samples
            freshness_acc = 100. * freshness_correct / total_samples
            combined_acc = (species_acc + freshness_acc) / 2
            
            # Update history
            history['train_species_loss'].append(train_species_loss)
            history['train_freshness_loss'].append(train_freshness_loss)
            history['val_species_loss'].append(val_species_loss)
            history['val_freshness_loss'].append(val_freshness_loss)
            history['val_species_acc'].append(species_acc)
            history['val_freshness_acc'].append(freshness_acc)
            
            print(f"  Train Species Loss: {train_species_loss:.4f} | Train Freshness Loss: {train_freshness_loss:.4f}")
            print(f"  Val Species Loss: {val_species_loss:.4f} | Val Freshness Loss: {val_freshness_loss:.4f}")
            print(f"  Val Species Acc: {species_acc:.2f}% | Val Freshness Acc: {freshness_acc:.2f}%")
            
            # Learning rate scheduling
            scheduler.step(val_species_loss + val_freshness_loss)
            
            # Save best model
            if combined_acc > best_val_acc:
                best_val_acc = combined_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'species_classes': num_species_classes,
                    'freshness_classes': num_freshness_classes,
                    'model_name': self.model_name,
                    'epoch': epoch,
                    'best_acc': best_val_acc,
                    'history': history
                }, 'best_fish_model.pt')
                print(f"  ✅ New best model saved! Combined accuracy: {best_val_acc:.2f}%")
        
        return model, history
    
    def evaluate_model(self, model, val_loader, species_encoder, freshness_encoder):
        """Comprehensive model evaluation"""
        print("\n🔍 Evaluating model...")
        
        model.eval()
        species_predictions = []
        freshness_predictions = []
        species_true = []
        freshness_true = []
        
        with torch.no_grad():
            for images, species_targets, freshness_targets in val_loader:
                images = images.to(self.device)
                species_outputs, freshness_outputs = model(images)
                
                _, species_pred = torch.max(species_outputs, 1)
                _, freshness_pred = torch.max(freshness_outputs, 1)
                
                species_predictions.extend(species_pred.cpu().numpy())
                freshness_predictions.extend(freshness_pred.cpu().numpy())
                species_true.extend(species_targets.numpy())
                freshness_true.extend(freshness_targets.numpy())
        
        # Calculate metrics
        species_acc = accuracy_score(species_true, species_predictions)
        freshness_acc = accuracy_score(freshness_true, freshness_predictions)
        
        species_f1 = f1_score(species_true, species_predictions, average='weighted')
        freshness_f1 = f1_score(freshness_true, freshness_predictions, average='weighted')
        
        print(f"\n📊 EVALUATION RESULTS:")
        print(f"  Species Classification - Accuracy: {species_acc:.4f}, F1-Score: {species_f1:.4f}")
        print(f"  Freshness Classification - Accuracy: {freshness_acc:.4f}, F1-Score: {freshness_f1:.4f}")
        
        # Classification reports
        print(f"\n🐟 SPECIES CLASSIFICATION REPORT:")
        # Get unique labels present in the validation set
        unique_species_labels = sorted(set(species_true))
        species_class_names = [species_encoder.classes_[i] for i in unique_species_labels]
        
        print(classification_report(species_true, species_predictions, 
                                  target_names=species_class_names,
                                  labels=unique_species_labels))
        
        print(f"\n🔄 FRESHNESS CLASSIFICATION REPORT:")
        unique_freshness_labels = sorted(set(freshness_true))
        freshness_class_names = [freshness_encoder.classes_[i] for i in unique_freshness_labels]
        
        print(classification_report(freshness_true, freshness_predictions,
                                  target_names=freshness_class_names,
                                  labels=unique_freshness_labels))
        
        # Confusion matrices
        self.plot_confusion_matrices(species_true, species_predictions, species_encoder.classes_,
                                   freshness_true, freshness_predictions, freshness_encoder.classes_)
        
        return {
            'species_accuracy': species_acc,
            'freshness_accuracy': freshness_acc,
            'species_f1': species_f1,
            'freshness_f1': freshness_f1
        }
    
    def plot_confusion_matrices(self, species_true, species_pred, species_classes,
                               freshness_true, freshness_pred, freshness_classes):
        """Plot confusion matrices for both tasks"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Get unique labels present in validation set
        unique_species_labels = sorted(set(species_true))
        unique_freshness_labels = sorted(set(freshness_true))
        
        species_class_names = [species_classes[i] for i in unique_species_labels]
        freshness_class_names = [freshness_classes[i] for i in unique_freshness_labels]
        
        # Species confusion matrix
        species_cm = confusion_matrix(species_true, species_pred, labels=unique_species_labels)
        sns.heatmap(species_cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=species_class_names, yticklabels=species_class_names, ax=ax1)
        ax1.set_title('Species Classification Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax1.get_yticklabels(), rotation=0)
        
        # Freshness confusion matrix
        freshness_cm = confusion_matrix(freshness_true, freshness_pred, labels=unique_freshness_labels)
        sns.heatmap(freshness_cm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=freshness_class_names, yticklabels=freshness_class_names, ax=ax2)
        ax2.set_title('Freshness Classification Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Confusion matrices saved as 'confusion_matrices.png'")

def main():
    """Main training pipeline"""
    print("🐟 FISH CLASSIFICATION MODEL TRAINING")
    print("=" * 60)
    
    # Dataset paths - CONFIGURE THESE TO YOUR LOCAL PATHS
    dataset_paths = {
        'fish_market': r'E:\sea\datasets\fish_market',
        'freshness': r'E:\sea\datasets\fresh and non-fresh fish',
        'fish_dataset': r'E:\sea\datasets\Fish dataset',
        'rohu': r'E:\sea\datasets\Rohu'
    }
    
    # Check if paths exist
    valid_dataset_paths = {}
    for name, path in dataset_paths.items():
        if os.path.exists(path):
            valid_dataset_paths[name] = path
        else:
            print(f"⚠️  Warning: {name} dataset path not found: {path}")
    
    if not valid_dataset_paths:
        print("❌ No dataset paths found! Please check your paths.")
        return
    
    # Step 1: Unify datasets
    unifier = DatasetUnifier()
    image_paths, species_labels, freshness_labels = unifier.unify_datasets(valid_dataset_paths)
    
    # Save encoders
    with open('species_encoder.pkl', 'wb') as f:
        pickle.dump(unifier.species_encoder, f)
    with open('freshness_encoder.pkl', 'wb') as f:
        pickle.dump(unifier.freshness_encoder, f)
    
    # Step 2: Create trainer and data loaders
    trainer = FishModelTrainer(model_name='resnet50', batch_size=32, learning_rate=0.001)
    train_loader, val_loader = trainer.create_data_loaders(image_paths, species_labels, freshness_labels)
    
    # Step 3: Train model
    num_species_classes = len(unifier.species_encoder.classes_)
    num_freshness_classes = len(unifier.freshness_encoder.classes_)
    
    print(f"\nTraining multi-task CNN:")
    print(f"  Species classes: {num_species_classes}")
    print(f"  Freshness classes: {num_freshness_classes}")
    
    model, history = trainer.train_model(train_loader, val_loader, num_species_classes, num_freshness_classes, epochs=30)
    
    # Step 4: Evaluate model
    metrics = trainer.evaluate_model(model, val_loader, unifier.species_encoder, unifier.freshness_encoder)
    
    # Step 5: Save final results
    results = {
        'training_completed': True,
        'model_path': 'best_fish_model.pt',
        'species_encoder_path': 'species_encoder.pkl',
        'freshness_encoder_path': 'freshness_encoder.pkl',
        'metrics': metrics,
        'species_classes': list(unifier.species_encoder.classes_),
        'freshness_classes': list(unifier.freshness_encoder.classes_),
        'total_samples': len(image_paths),
        'dataset_paths': valid_dataset_paths
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"✅ Model saved: best_fish_model.pt")
    print(f"✅ Encoders saved: species_encoder.pkl, freshness_encoder.pkl")
    print(f"✅ Results saved: training_results.json")
    print(f"✅ Total samples trained: {len(image_paths):,}")
    print(f"✅ Final accuracy - Species: {metrics['species_accuracy']:.2%}, Freshness: {metrics['freshness_accuracy']:.2%}")

if __name__ == "__main__":
    main()

