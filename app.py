# Import neccessary libraries

import os
import cv2 as cv
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load model
model = load_model("Models/Brain_cancer_model.h5")

# Initialize Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Home page
@app.route('/')
def home():
    return render_template('home.html')


# About page
@app.route('/about')
def about():
    return render_template('about.html')


# Predict page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        file = request.files.get('file')  # 'file' not 'image', to match fetch()

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                
                # Run actual model prediction
                img = Image.open(filepath).resize((150, 150))
                image = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                image = image.reshape(1, 150,150, 3)
                
                # Predict using model
                pred = model.predict(image)
                pred_class = np.argmax(pred, axis=1)[0]
                
                # Map prediction to class category
                labels = {
                    0: "Glioma Tumor",
                    1: "No Tumor",
                    2: "Meningioma Tumor",
                    3: "Pituitary Tumor"
                }
                    

                
                prediction = labels.get(pred_class, "Unknown")
                predictionClass = "danger" if pred_class != 1 else "healthy"
                description = (
                    f"The scan indicates presence of a {label.lower()}." if pred_class != 1
                else "No tumor was detected in the brain scan.")
                    
                result = {
                    "prediction": prediction,
                    "predictionClass": predictionClass,
                    "description": description
                }
                
                
                return jsonify(result)

                # # Or render to HTML page
                # return render_template('results.html', result=result, image_url=filepath)


            except Exception as e:
                return jsonify({"error": str(e)}), 500
        else:
            return jsonify({"error": "Invalid file type. Please upload a PNG or JPEG image."}), 400
    else:
        return render_template('predict.html')


# Results page 
@app.route('/results', methods=['GET'])
def results():
    return render_template('results.html')

# Delete last uploaded file
@app.after_request
def remove_uploaded_file(response):
    try:
    # Only delete file after POST to /predict
        if request.endpoint == 'predict' and request.method == 'POST':
            file = request.files.get('file')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
    except Exception:
        pass
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
