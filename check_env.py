import keras
import torch
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not installed")

print(f"Keras version: {keras.__version__}")
print(f"Torch version: {torch.__version__}")
print(f"Keras backend: {keras.backend.backend()}")
