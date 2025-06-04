# Import neccessary libraries

import os
import cv2 as cv
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")


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
                img = img.convert('RGB')
                # Convert image to numpy array and preprocess
                image = np.array(img)
                image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
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
               
                
                prediction = labels.get(pred_class, "")
                print("Prediction:", prediction)
                
                # For classification in JS
                label = ["glioma", "healthy", "meningioma", "pituitary"]

                
                if pred_class != 1:
                    predictionClass = label[pred_class] 
                else:
                    predictionClass = "healthy"
                
                print("PredictionClass:", predictionClass)
                
                # Prepare description based on prediction
                if pred_class != 1:
                    description = (
                        f"Our analysis suggests the presence of a {prediction.lower()} in the brain scan. "
                        "We recommend consulting a medical professional for a comprehensive diagnosis and further guidance.")
                    
                    if pred_class == 0:
                            description  += " Glioma tumors are a type of tumor that occurs in the brain and spinal cord. "
                    elif pred_class == 2:
                            description  += " Meningioma tumors are typically benign tumors that arise from the meninges, the protective membranes covering the brain and spinal cord. "
                    elif pred_class == 3:
                            description  += " Pituitary tumors are abnormal growths that develop in the pituitary gland, a small gland located at the base of the brain. "
                
                else:
                    description = "No tumor was detected in the brain scan."
                    description  += " The brain scan appears to be normal, with no signs of tumors detected."
                    
                result = {
                    "prediction": prediction,
                    "predictionClass": predictionClass,
                    "description": description,
                    "image_url": url_for('static', filename='uploads/' + filename)
                }
                
                return jsonify(result)

            except Exception as e:
                return jsonify({"error": str(e)}), 500
        else:
            return jsonify({"error": "Invalid file type. Please upload a PNG or JPEG image."}), 400
    else:
        return render_template('predict.html')



# About page
@app.route('/about')
def about():
    return render_template('about.html')


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
    app.run(debug=True)
