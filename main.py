from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__,
            template_folder='templates',  # Set the template folder explicitly (optional if the folder is named 'templates')
            static_folder='static')  

# Load the saved models
try:
    cnn_model = load_model("cnn_model.h5")
    hybrid_model = load_model("hybrid_model.h5")
    transfer_model = load_model("transfer_model.h5")
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Directory to store uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

# Render other pages
@app.route('/CNN1.html')
def cnn1():
    return render_template('CNN1.html')

@app.route('/CNN2.html')
def cnn2():
    return render_template('CNN2.html')

@app.route('/Hybrid1.html')
def hybrid1():
    return render_template('Hybrid1.html')

@app.route('/Hybrid2.html')
def hybrid2():
    return render_template('Hybrid2.html')

@app.route('/Transfer1.html')
def transfer1():
    return render_template('Transfer1.html')

@app.route('/Transfer2.html')
def transfer2():
    return render_template('Transfer2.html')

@app.route('/play.html')
def play():
    return render_template('play.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    try:
        img_array = preprocess_image(filepath, target_size=(128, 128))
        
        # Get predictions from all models
        cnn_prediction = get_prediction(cnn_model, img_array)
        hybrid_prediction = get_prediction(hybrid_model, img_array)
        transfer_prediction = get_prediction(transfer_model, img_array)
        
        response = {
            "cnn": cnn_prediction,
            "hybrid": hybrid_prediction,
            "transfer": transfer_prediction
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Function to preprocess image
def preprocess_image(img_path, target_size):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to get prediction from a model
def get_prediction(model, img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    class_names = ["Mild Dementia", "Moderate Dementia", "No Dementia", "Very Mild Dementia"]
    return class_names[predicted_class]

if __name__ == '__main__':
    app.run(debug=True)
