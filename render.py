# Import libraries

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


app = Flask(__name__)

# # Load model on startup
model = load_model("Models/Brain_cancer_model.h5")

# Preprocessing image
def preprocess_image(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Predict route
@app.route('/predict', methods=['POST'])
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

@app.route('/')
def index():
    return "Brain Cancer Prediction API is running."

if __name__ == '__main__':
    app.run(debug=True)
