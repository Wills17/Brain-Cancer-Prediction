# Import libraries

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


api_app = Flask(__name__)

# for debugging purpose
import os
print("Current working directory:", os.getcwd())
print("Files:", os.listdir())
print("Models folder:", os.listdir("Models"))


# Load model on startup
try:
    model_path = os.path.join(os.path.dirname(__file__), "Models", "Brain_cancer_model.h5")
    model = load_model(model_path)
except Exception as e:
    print("Error loading model:", e)
    raise

# Preprocessing image
def preprocess_image(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Predict route
@api_app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
        
        image = preprocess_image(img)
        prediction = model.predict(image)
        
        pred_class = np.argmax(prediction, axis=1)[0]
        
        labels = ["glioma_tumor", "no_tumor", "meningioma_tumor", "pituitary_tumor"]
        label = labels[pred_class]

        response = {
            "prediction": label,
            "predictionClass": label,
            "description": f"{label.capitalize()} tumor detected." if label != 'no_tumor' else "No tumor detected in the scan."
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_app.route('/')
def index():
    return "Brain Cancer Prediction API is running."

if __name__ == '__main__':
    api_app.run(debug=True)
