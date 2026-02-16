import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def get_data_loaders(data_dir, batch_size=32, image_size=(224, 224)):
    """
    Creates PyTorch DataLoaders for train, validation, and test sets.
    This serves as the PyTorch alternative to TensorFlow's ImageDataGenerator.
    """
    
    # Define transformations for training (with data augmentation)
    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Define transformations for validation and testing (no augmentation, just resize and normalize)
    val_test_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    loaders = {}
    
    # Load datasets using ImageFolder
    for split in ['train', 'validation', 'test']:
        split_path = os.path.join(data_dir, split)
        
        if not os.path.exists(split_path):
            print(f"Warning: Split path '{split_path}' not found. Skipping.")
            continue
            
        transform = train_transforms if split == 'train' else val_test_transforms
        dataset = datasets.ImageFolder(root=split_path, transform=transform)
        
        shuffle = True if split == 'train' else False
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
        
        print(f"Loaded {len(dataset)} images for '{split}' across {len(dataset.classes)} classes.")
        
    return loaders

def imshow(img, title=None):
    """
    Helper function to display an image (un-normalizing first).
    """
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis('off')

def visualize_batch(dataloader, class_names):
    """
    Visualizes a batch of preprocessed images.
    """
    images, labels = next(iter(dataloader))
    
    plt.figure(figsize=(16, 8))
    for i in range(min(8, len(images))):
        plt.subplot(2, 4, i + 1)
        imshow(images[i], title=class_names[labels[i]])
    
    plt.tight_layout()
    plt.savefig('preprocessed_sample.png')
    print("Sample batch visualization saved as 'preprocessed_sample.png'")

if __name__ == "__main__":
    DATASET_DIR = "dataset"
    BATCH_SIZE = 32
    IMAGE_SIZE = (224, 224)
    
    # Initialize Loaders
    data_loaders = get_data_loaders(DATASET_DIR, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
    
    # Quick Check
    if 'train' in data_loaders:
        train_loader = data_loaders['train']
        class_names = train_loader.dataset.classes
        print(f"\nClass names: {class_names}")
        
        # Visualize the first batch
        print("Displaying a sample batch of preprocessed training images...")
        visualize_batch(train_loader, class_names)
