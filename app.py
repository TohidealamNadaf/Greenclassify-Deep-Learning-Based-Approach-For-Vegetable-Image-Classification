import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

# Set Keras backend to torch before importing keras
os.environ["KERAS_BACKEND"] = "torch"
import keras

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and classes
model = None
class_names = []

def load_prediction_model():
    global model, class_names
    try:
        # Load class names
        if os.path.exists('class_names.json'):
            with open('class_names.json', 'r') as f:
                class_names = json.load(f)
        else:
            print("Warning: class_names.json not found.")
        
        # Load model - prioritizing the final fixed Xception model
        model_path = 'xception_v4_final.keras'
        if not os.path.exists(model_path):
             model_path = 'xception_model_fixed.keras'
        if not os.path.exists(model_path):
             model_path = 'interrupted_model.keras'
        if not os.path.exists(model_path):
            model_path = 'vegetable_resnet50_full.keras'
            
        if os.path.exists(model_path):
            try:
                # Attempt to load the model. 
                # Note: H5 models from older Keras versions might need compile=False in Keras 3
                model = keras.models.load_model(model_path, compile=False)
                print(f"Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
                # Fallback logic
                if 'xception' in model_path:
                    print("Attempting to load fallback ResNet model...")
                    alt_path = 'vegetable_resnet50_full.keras'
                    if os.path.exists(alt_path):
                        model = keras.models.load_model(alt_path)
                        print(f"Fallback model loaded: {alt_path}")
        else:
            print("Warning: No model file found. Predictions will not work.")
    except Exception as e:
        print(f"Error in load_prediction_model: {e}")

# Initial model load
load_prediction_model()

def preprocess_image(image_path):
    """
    Load and preprocess image based on model's expected input shape.
    """
    img = Image.open(image_path).convert('RGB')
    
    # Default input size
    input_size = (224, 224)
    
    if model:
        try:
            # Dynamically get input size from model
            shape = model.input_shape
            # If it's a list (multiple inputs), take the first one
            if isinstance(shape, list): shape = shape[0]
            # shape is usually (None, height, width, channels)
            if shape and len(shape) >= 3:
                input_size = (shape[1], shape[2])
        except Exception as e:
            print(f"Warning: Could not determine model input shape: {e}")
    
    img = img.resize(input_size)
    img_array = np.array(img).astype('float32')
    
    # Apply normalization based on model type/size
    if input_size == (299, 299):
        # Xception-style normalization: [-1, 1]
        img_array = (img_array / 127.5) - 1.0
    else:
        # ResNet/Default normalization: [0, 1]
        img_array = img_array / 255.0  
        
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        if not model:
            return "Model not loaded. Please ensure a .keras or .h5 model exists in the project root.", 500
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Preprocess and Predict
            processed_img = preprocess_image(filepath)
            predictions = model.predict(processed_img)
            
            # If the model ends with a linear layer (logits), apply softmax
            # (Checking for 'linear' activation in the last layer or just applying it to be safe)
            if hasattr(model.layers[-1], 'activation') and model.layers[-1].activation.__name__ == 'linear':
                # Apply softmax manually to get probabilities
                exp_preds = np.exp(predictions[0] - np.max(predictions[0])) # stability
                predictions = [exp_preds / exp_preds.sum()]
            
            score = np.max(predictions[0])
            class_idx = np.argmax(predictions[0])
            label = class_names[class_idx] if class_idx < len(class_names) else "Unknown"
            
            return render_template('prediction.html', 
                                 label=label, 
                                 confidence=f"{score*100:.2f}%",
                                 image_url=filename) 
        except Exception as e:
            return f"Prediction error: {str(e)}", 500
    
    return "Invalid request", 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
