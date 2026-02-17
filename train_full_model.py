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
import time
import multiprocessing

# Optimization: Check GPU availability upfront
def check_gpu():
    print("-" * 30)
    print("DEVICE DIAGNOSTICS")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU detected. Training will be extremely slow on CPU.")
    print("-" * 30)

def to_nhwc(x):
    """
    Top-level function to convert NCHW to NHWC for pickling support on Windows.
    """
    return x.permute(1, 2, 0)

class DiagnosticCallback(callbacks.Callback):
    """
    Custom callback for detailed logging of the training process.
    """
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n[DIAGNOSTIC] Starting Epoch {epoch + 1} at {time.ctime()}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"[DIAGNOSTIC] Finished Epoch {epoch + 1} at {time.ctime()}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[DIAGNOSTIC] GPU cache cleared.")

    def on_test_begin(self, logs=None):
        print(f"[DIAGNOSTIC] Starting Validation phase at {time.ctime()}")

    def on_test_end(self, logs=None):
        print(f"[DIAGNOSTIC] Finished Validation phase at {time.ctime()}")

# Removed custom data_generator as Keras 3 fit() supports PyTorch DataLoaders directly

def get_data_loaders(data_dir, batch_size=32, image_size=(224, 224)):
    """
    Creates DataLoaders and wraps them in generators.
    """
    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        # Convert NCHW to NHWC for Keras defaults
        transforms.Lambda(to_nhwc) # Use top-level function for pickling support
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        # Convert NCHW to NHWC for Keras defaults
        transforms.Lambda(to_nhwc) # Use top-level function for pickling support
    ])

    loaders = {}
    steps = {}
    class_names = None
    
    for split in ['train', 'validation']:
        path = os.path.join(data_dir, split)
        if not os.path.exists(path):
            print(f"Directory {path} does not exist. Skipping.")
            continue
            
        dataset = datasets.ImageFolder(root=path, transform=(train_transforms if split == 'train' else val_transforms))
        
        if split == 'train':
            class_names = dataset.classes
            
        # Parallel data loading (using num_workers=2 for a balance between overhead and speed)
        num_workers = 0
        
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=(split == 'train'), 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        loaders[split] = loader
        steps[split] = len(loader)
        
        print(f"Loaded {len(dataset)} images for '{split}' ({steps[split]} steps per epoch, num_workers={num_workers}).")
        
    return loaders, steps, class_names

def build_resnet_model(num_classes):
    """
    Builds the ResNet50 transfer learning model.
    """
    # Use Keras 3 ResNet50 implementation
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
    # Reduced epochs for re-verification run
    EPOCHS = 10 
    MODEL_NAME = "vegetable_resnet50_full.keras"

    # 0. Diagnostics
    check_gpu()

    # 1. Load Data
    data_loaders, steps, class_names = get_data_loaders(DATA_DIR, BATCH_SIZE)
    if not class_names:
        print("Failed to load dataset. Exiting.")
        exit()
        
    with open("class_names.json", 'w') as f:
        json.dump(class_names, f)

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
        DiagnosticCallback(),
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(filepath=MODEL_NAME, monitor='val_accuracy', save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]

    # 4. Train
    print("\nStarting Full Training with Diagnostic Logging...")
    try:
        history = model.fit(
            data_loaders['train'],
            validation_data=data_loaders['validation'],
            epochs=EPOCHS,
            callbacks=my_callbacks,
            verbose=1
        )
        
        model.save(MODEL_NAME)
        print(f"\nTraining complete. Model saved as '{MODEL_NAME}'")
        plot_history(history)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
