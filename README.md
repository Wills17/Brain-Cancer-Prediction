# Brain Cancer Prediction

This project uses deep learning to classify brain MRI images into four categories: Glioma Tumor, Meningioma Tumor, Pituitary Tumor, and No Tumor. It includes model training, prediction scripts, and a modern Flask web application for user-friendly image uploads and instant AI-powered predictions.

## Project Structure

```
app.py
app_r.py
brain_cancer_code.py
brain_cancer_predict.py
render.py
render.yaml
requirements.txt
README.md
Dataset.zip
Models/
    Brain_cancer_model.h5
    Brain_cancer_model.tflite
static/
    scripts.js
    styles.css
    uploads/
templates/
    about.html
    home.html
    predict.html
    results.html
```

- **app.py**: Main Flask web app for uploading MRI images and displaying predictions using the `.h5` model.
- **app_r.py**: Alternative Flask app using the TensorFlow Lite (`.tflite`) model for lightweight inference.
- **brain_cancer_code.py**: Model training and evaluation using EfficientNetB0 and Keras.
- **brain_cancer_predict.py**: Script for visual/manual prediction from images in the `Dataset/Predict/` folder.
- **render.py, render.yaml**: Deployment configuration files (e.g., for Render.com).
- **requirements.txt**: Python dependencies.
- **Models/**: Stores trained model files and conversion scripts.
- **static/**: Frontend static files (JS, CSS, uploads).
- **templates/**: HTML templates for the web interface.

## Getting Started

### Prerequisites

- Python 3.7+
- TensorFlow
- Keras
- OpenCV
- Flask
- NumPy
- Matplotlib
- scikit-learn
- tqdm
- seaborn
- Pillow

Install dependencies:
```sh
pip install -r requirements.txt
```

### Dataset

Organize your dataset as follows:
- `Dataset/Training/` — Training images, separated by class folders.
- `Dataset/Testing/` — Testing images, separated by class folders.
- `Dataset/Predict/` — Images for prediction.

### Training the Model

To train the model and save it:
```sh
python brain_cancer_code.py
```
This will preprocess the data, train the model, and save it as `Models/Brain_cancer_model.h5`.

### Model Conversion (Optional)

To convert the trained model to TensorFlow Lite:
```sh
python Models/convert.py
```
This will generate `Models/Brain_cancer_model.tflite`.

### Making Predictions

#### Visual Prediction Script

To predict a random image from the `Dataset/Predict/` folder:
```sh
python brain_cancer_predict.py
```
You will be prompted to confirm the image before prediction.

#### Web Application

To run the Flask web app (standard Keras model):
```sh
python app.py
```
Or to use the TensorFlow Lite version:
```sh
python app_r.py
```
- Open your browser and go to `http://127.0.0.1:5000/`
- Upload an MRI image to get a prediction.

## Features

- Drag & drop or browse to upload MRI scans.
- Instant AI-powered predictions with visual feedback.
- Downloadable analysis report.
- Responsive and modern UI.
- Privacy-focused: images are not stored after analysis.

## Notes

- The model uses EfficientNetB0 as a feature extractor.
- Make sure the `Models/Brain_cancer_model.h5` (or `.tflite`) file exists before running prediction scripts or the web app.
- For deployment, see `render.py` and `render.yaml`.

---