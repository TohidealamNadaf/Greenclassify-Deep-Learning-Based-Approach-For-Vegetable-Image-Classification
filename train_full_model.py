import os

# Set Keras backend to PyTorch BEFORE importing keras
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers, models, callbacks
from keras.applications import ResNet50
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json

def get_data_loaders(data_dir, batch_size=32, image_size=(224, 224)):
    """
    Creates DataLoaders for all classes in the dataset.
    Ensures NHWC format for Keras 3 compatibility.
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
    class_names = None
    
    for split in ['train', 'validation']:
        path = os.path.join(data_dir, split)
        if not os.path.exists(path):
            print(f"Directory {path} does not exist. Skipping.")
            continue
            
        dataset = datasets.ImageFolder(root=path, transform=(train_transforms if split == 'train' else val_transforms))
        
        if split == 'train':
            class_names = dataset.classes
            
        # We use num_workers=0 to avoid potential multiprocessing issues in some environments
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), num_workers=0)
        print(f"Loaded {len(dataset)} images for '{split}' (Classes: {len(dataset.classes)})")
        
    return loaders, class_names

def build_resnet_model(num_classes):
    """
    Builds the ResNet50 transfer learning model.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_history(history):
    """
    Plots the training and validation accuracy and loss.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")

if __name__ == "__main__":
    DATA_DIR = "dataset"
    BATCH_SIZE = 32
    EPOCHS = 20 # Increased for full training
    MODEL_NAME = "vegetable_resnet50_full.keras"
    CLASS_NAMES_FILE = "class_names.json"

    # 1. Load Data
    data_loaders, class_names = get_data_loaders(DATA_DIR, BATCH_SIZE)
    if not class_names:
        print("Failed to load dataset. Exiting.")
        exit()
        
    # Save class names for future inference
    with open(CLASS_NAMES_FILE, 'w') as f:
        json.dump(class_names, f)
    print(f"Class names saved to {CLASS_NAMES_FILE}")

    # 2. Build and Compile Model
    model = build_resnet_model(len(class_names))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 3. Callbacks
    my_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(filepath=MODEL_NAME, monitor='val_accuracy', save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]

    # 4. Train
    print("\nStarting Full Training...")
    try:
        history = model.fit(
            data_loaders['train'],
            validation_data=data_loaders['validation'],
            epochs=EPOCHS,
            callbacks=my_callbacks
        )
        
        # 5. Save Final Model (Already handled by Checkpoint, but good to be explicit)
        model.save(MODEL_NAME)
        print(f"\nTraining complete. Model saved as '{MODEL_NAME}'")
        
        # 6. Plot History
        plot_history(history)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")
        model.save("interrupted_model.keras")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
