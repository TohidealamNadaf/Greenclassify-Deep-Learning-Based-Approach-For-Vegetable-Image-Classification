import os

# Set Keras backend to PyTorch BEFORE importing keras
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers, models
from keras.applications import ResNet50
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def get_subset_loaders(data_dir, selected_classes, batch_size=32, image_size=(224, 224)):
    """
    Creates DataLoaders for a subset of classes.
    """
    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        # Convert NCHW to NHWC for Keras defaults
        transforms.Lambda(lambda x: x.permute(1, 2, 0).numpy()) 
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        # Convert NCHW to NHWC for Keras defaults
        transforms.Lambda(lambda x: x.permute(1, 2, 0).numpy())
    ])

    loaders = {}
    for split in ['train', 'validation']:
        path = os.path.join(data_dir, split)
        if not os.path.exists(path):
            print(f"Directory {path} does not exist.")
            continue
            
        full_dataset = datasets.ImageFolder(root=path)
        
        # Filter indices
        class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}
        selected_indices = [i for i, (_, label_idx) in enumerate(full_dataset.samples) 
                            if full_dataset.classes[label_idx] in selected_classes]
        
        # Create a mapping for labels
        class_mapping = {full_dataset.class_to_idx[cls]: class_to_idx[cls] for cls in selected_classes}
        
        # Custom Dataset class to handle the subset and transform
        class SubsetDataset(torch.utils.data.Dataset):
            def __init__(self, full_dataset, indices, mapping, transform):
                self.full_dataset = full_dataset
                self.indices = indices
                self.mapping = mapping
                self.transform = transform
            def __getitem__(self, idx):
                img, label = self.full_dataset[self.indices[idx]]
                if self.transform:
                    img = self.transform(img)
                return img, self.mapping[label]
            def __len__(self):
                return len(self.indices)

        dataset = SubsetDataset(full_dataset, selected_indices, class_mapping, 
                                 (train_transforms if split == 'train' else val_transforms))
        
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), num_workers=0)
        print(f"Loaded {len(dataset)} images for '{split}' subset (Classes: {selected_classes})")
        
    return loaders

def create_resnet_transfer_model(num_classes):
    """
    Creates a ResNet50 transfer learning model.
    """
    # Load ResNet50 pre-trained on ImageNet, without the top layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

if __name__ == "__main__":
    DATASET_DIR = "dataset"
    SELECTED_CLASSES = ['Bean', 'Brinjal', 'Carrot', 'Potato', 'Tomato']
    BATCH_SIZE = 32
    EPOCHS = 5 # Small number for demonstration
    
    print(f"Using Keras with Backend: {keras.config.backend()}")
    
    # 1. Prepare Data Loaders (Subset of 5 classes)
    loaders = get_subset_loaders(DATASET_DIR, SELECTED_CLASSES, batch_size=BATCH_SIZE)
    
    # 2. Build Model
    model = create_resnet_transfer_model(len(SELECTED_CLASSES))
    
    # 3. Compile Model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy', # Using sparse since labels are integers 0-4
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 4. Training (Simple loop or Keras fit using PyTorch DataLoaders)
    # Keras 3 fit() can take PyTorch DataLoaders directly if the backend is 'torch'
    print("\nStarting Training on 5-class subset...")
    
    try:
        # We only run for a couple of batches or 1 epoch for verification
        model.fit(
            loaders['train'],
            validation_data=loaders['validation'],
            epochs=1 # Demonstration run
        )
        
        # Save the model
        model.save('vegetable_resnet50_5class.keras')
        print("\nModel saved as 'vegetable_resnet50_5class.keras'")
        
    except Exception as e:
        print(f"An error occurred during training: {e}")
