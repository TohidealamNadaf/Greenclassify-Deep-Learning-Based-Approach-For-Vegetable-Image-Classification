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
        
        # Load model - checking for the best available model file
        model_path = 'interrupted_model.keras'
        if not os.path.exists(model_path):
            model_path = 'vegetable_resnet50_full.keras'
            
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        else:
            print("Warning: No model file found. Predictions will not work.")
    except Exception as e:
        print(f"Error loading model: {e}")

# Initial model load
load_prediction_model()

def preprocess_image(image_path):
    """
    Load and preprocess image for ResNet50 (224x224).
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    # Scale to [0, 1] as per common training practices
    img_array = img_array.astype('float32') / 255.0  
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
            return "Model not loaded. Please ensure a .keras model exists in the project root.", 500
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Preprocess and Predict
            processed_img = preprocess_image(filepath)
            predictions = model.predict(processed_img)
            
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
