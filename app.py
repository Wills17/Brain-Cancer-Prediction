from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import requests
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# External prediction API endpoint (Replace with your actual Render API endpoint)
RENDER_API_URL = 'https://brain-cancer-api.onrender.com/predict'

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


# Predict page (image upload form)
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('image')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Send image to Render API
            with open(filepath, 'rb') as f:
                files = {'file': (filename, f, file.content_type)}
                try:
                    response = requests.post(RENDER_API_URL, files=files)
                    if response.status_code == 200:
                        result = response.json()
                        return render_template('results.html', result=result, image_url=filepath)
                    else:
                        return render_template('results.html', error="Prediction API error", image_url=filepath)
                except Exception as e:
                    return render_template('results.html', error=str(e), image_url=filepath)
        else:
            return render_template('predict.html', error="Invalid file type. Please upload a PNG or JPG.")
    return render_template('predict.html')


# Results page (uses result passed from predict route)
@app.route('/results', methods=['GET'])
def results():
    return render_template('results.html')


# API fallback (optional for local testing)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    return jsonify({
        "prediction": "No Tumor",
        "predictionClass": "healthy",
        "description": "No tumor was detected in the brain scan."
    })


if __name__ == '__main__':
    app.run(debug=True)
