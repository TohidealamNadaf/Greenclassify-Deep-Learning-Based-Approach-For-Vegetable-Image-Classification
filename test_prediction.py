import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import numpy as np
from PIL import Image
import json

def test_prediction(model_path, image_path, class_names_path):
    print(f"Testing model: {model_path}")
    try:
        # Load class names
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        
        # Load model
        model = keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully.")
        
        # Preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((299, 299))
        img_array = np.array(img).astype('float32') 
        # Xception-style normalization: [-1, 1]
        img_array = (img_array / 127.5) - 1.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array)
        
        # Apply softmax if linear
        if hasattr(model.layers[-1], 'activation') and model.layers[-1].activation.__name__ == 'linear':
            exp_preds = np.exp(predictions[0] - np.max(predictions[0]))
            predictions = [exp_preds / exp_preds.sum()]
            
        score = np.max(predictions[0])
        class_idx = np.argmax(predictions[0])
        label = class_names[class_idx] if class_idx < len(class_names) else "Unknown"
        
        print(f"Prediction for {image_path}: {label} (Confidence: {score*100:.2f}%)")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Find a sample image
    sample_img = None
    dataset_dir = "dataset/validation"
    if os.path.exists(dataset_dir):
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    sample_img = os.path.join(root, file)
                    break
            if sample_img: break
    
    if sample_img:
        test_prediction("xception_v4_final.keras", sample_img, "class_names.json")
    else:
        print("No sample image found in dataset/validation")
