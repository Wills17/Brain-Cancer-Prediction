# Brain Cancer Prediction

This project uses deep learning to classify brain MRI images into four categories: Glioma Tumor, Meningioma Tumor, Pituitary Tumor, and No Tumor. It includes model training, prediction scripts, and a Flask web application for user-friendly image uploads and predictions.

## Project Structure

```
app.py
brain_cancer_code.py
brain_cancer_predict.py
README.md
Dataset/
    Predict/
    Testing/
    Training/
Models/
    Brain_cancer_model.h5
    EfficientNetB0_model.h5
```

- **app.py**: Flask web app for uploading MRI images and displaying predictions.
- **brain_cancer_code.py**: Model training and evaluation using EfficientNetB0 and Keras.
- **brain_cancer_predict.py**: Script for visual/manual prediction from images in the Predict folder.
- **Dataset/**: Contains subfolders for training, testing, and prediction images.
- **Models/**: Stores trained model files.

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
pip install tensorflow keras opencv-python flask numpy matplotlib scikit-learn tqdm seaborn pillow
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

### Making Predictions

#### Visual Prediction Script

To predict a random image from the `Predict` folder:
```sh
python brain_cancer_predict.py
```
You will be prompted to confirm the image before prediction.

#### Web Application

To run the Flask web app:
```sh
python app.py
```
- Open your browser and go to `http://127.0.0.1:5000/`
- Upload an MRI image to get a prediction.

## Notes

- The model uses EfficientNetB0 as a feature extractor.
- Make sure the `Models/Brain_cancer_model.h5` file exists before running prediction scripts or the web app.