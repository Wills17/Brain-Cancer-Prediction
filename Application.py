# Import neccessary libraries

import os
import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# Build Flask app
app = Flask(__name__)

# app.config["UPLOAD_FOLDER"] = "User_Upload"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load the model
model = load_model("Models/Brain_cancer_model.h5")

# State file formats
extensions = {"jpg", "jpeg", "png"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in extensions


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET"])
def predict_form():
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("predict.html", error="No file part")
        file = request.files["file"]
        
        if file.filename == "":
            return render_template("home.html", error="No selected file")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            return render_template("results.html", filename=filename, prediction=prediction) 
        
        else:
            return render_template("home.html", error="Invalid file type")
    return render_template("home.html")


@app.route("/results")
def results():
    return render_template("results.html")


@app.route('/about')
def about():
    return render_template('about.html')


# @app.route("/uploads/<filename>")
# def uploaded_file(filename):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)


# End