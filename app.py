# Import neccessary libraries
from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Build Flask app
app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "User_Upload"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load the model
model = load_model("Models/Brain_cancer_model.h5")

extensions = {"jpg", "jpeg", "png"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in extensions

def predict_image(filepath):
    image = cv.imread(filepath)
    image = cv.resize(image, (150, 150))
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    image = image.reshape(1, 150, 150, 3)
    pred = model.predict(image)
    pred = np.argmax(pred, axis=1)[0]
    if pred == 0:
        return "Glioma Tumor"
    elif pred == 1:
        return "No Tumor"
    elif pred == 2:
        return "Meningioma Tumor"
    else:
        return "Pituitary Tumor"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file part")
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No selected file")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            return render_template("result.html", filename=filename, prediction=prediction)
        else:
            return render_template("index.html", error="Invalid file type")
    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return redirect(url_for("static", filename="User_Upload/" + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)


# End