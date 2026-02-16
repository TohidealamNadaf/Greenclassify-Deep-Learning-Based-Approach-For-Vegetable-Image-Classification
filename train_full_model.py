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

def data_generator(dataloader):
    """
    Wraps a DataLoader into a persistent generator.
    Yields (images, labels) as numpy arrays in HWC format.
    """
    while True:
        for images, labels in dataloader:
            # images: (Batch, H, W, C) numpy array
            # labels: (Batch,) numpy array
            yield images, labels

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
        transforms.Lambda(lambda x: x.permute(1, 2, 0).numpy()) 
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        # Convert NCHW to NHWC for Keras defaults
        transforms.Lambda(lambda x: x.permute(1, 2, 0).numpy())
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
            
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), num_workers=0)
        loaders[split] = data_generator(loader)
        steps[split] = int(np.ceil(len(dataset) / batch_size))
        
        print(f"Loaded {len(dataset)} images for '{split}' ({steps[split]} steps per epoch).")
        
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

    # 1. Load Data
    data_gens, steps, class_names = get_data_loaders(DATA_DIR, BATCH_SIZE)
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
            data_gens['train'],
            steps_per_epoch=steps['train'],
            validation_data=data_gens['validation'],
            validation_steps=steps['validation'],
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
