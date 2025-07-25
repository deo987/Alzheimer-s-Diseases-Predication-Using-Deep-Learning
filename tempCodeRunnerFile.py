from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the saved CNN model
try:
    cnn_model = load_model("newmodel.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Directory to store uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route to serve the HTML file
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file to the upload folder
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess and predict the image
    try:
        img_array = preprocess_image(filepath, target_size=(128, 128))  # Updated target_size
        predictions = cnn_model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        class_names = ["Mild Impairment", "Moderate Impairment", "No Impairment", "Very Mild Impairment"]  # Updated class labels
        predicted_label = class_names[predicted_class]
        return jsonify({'prediction': predicted_label, 'confidence': float(confidence), 'probabilities': predictions[0].tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Image preprocessing function
def preprocess_image(img_path, target_size):
    img = Image.open(img_path).convert('RGB')  # Ensure RGB
    img = img.resize(target_size)  # Resize to match model's input size
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

if __name__ == '__main__':
    app.run(debug=True)