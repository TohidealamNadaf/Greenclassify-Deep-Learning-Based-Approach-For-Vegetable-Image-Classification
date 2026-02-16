
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def setup_data_generators(dataset_dir, batch_size=32, img_size=(224, 224)):
    """
    Sets up ImageDataGenerators for training, validation, and testing.
    """
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs detected: {gpus}")
    else:
        print("No GPUs detected. Running on CPU.")

    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'validation')
    test_dir = os.path.join(dataset_dir, 'test')

    # Data Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Rescaling for validation and testing (no augmentation)
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    print("\nLoading Training Data...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    print("\nLoading Validation Data...")
    validation_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    print("\nLoading Test Data...")
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False 
    )

    return train_generator, validation_generator, test_generator

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    BATCH_SIZE = 32
    IMG_SIZE = (224, 224)

    try:
        train_gen, val_gen, test_gen = setup_data_generators(dataset_dir, BATCH_SIZE, IMG_SIZE)
        
        # Verify generators
        print("\n--- Data Generator Verification ---")
        print(f"Class Indices: {train_gen.class_indices}")
        
        # Check a batch
        x_batch, y_batch = next(train_gen)
        print(f"Batch Shape (X): {x_batch.shape}")
        print(f"Batch Shape (Y): {y_batch.shape}")
        print("Data preprocessing setup successful.")

    except Exception as e:
        print(f"An error occurred: {e}")
