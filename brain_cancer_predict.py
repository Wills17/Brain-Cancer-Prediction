"""For visual predictions"""

# Import neccessary libraries
import os
import random
import cv2 as cv
import numpy as np
import warnings
from tensorflow.keras.models import load_model 
warnings.filterwarnings("ignore")


# Allowed extensions
extensions = ["jpg", "jpeg", "png"]

# Pick random folder and image in folder
image_path = "Dataset/Predict/"
    
image_picked = random.choice(os.listdir(image_path))
print("\nImage picked:", image_picked)
    
# Check image extension
if image_picked.split(".")[-1].lower() not in extensions:
    print(f"Image format is invalid: {image_picked.split(".")[-1]}")
    print("Supported extensions are: {}".format(extensions))

# Update image path
new_image_path = image_path + "/" + image_picked
print("\nNew image path:", new_image_path)

# Read and process image
image = cv.imread(new_image_path)

# Display original image
cv.imshow(f"Original Image: {image_picked}", image)
cv.waitKey(10000)
        
# Resize and display the resized image
image = cv.resize(image, (150, 150))
cv.imshow(f"Resized Image:{image_picked}", image)
cv.waitKey(10000)
cv.destroyAllWindows()
    
# Convert into numpy array
image = np.array(image)

image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
image = image.reshape(1,150,150,3)



# === STEP 1: Use MRI Classifier to check if this is a brain MRI ===
print("\nChecking if uploaded image is a brain MRI...")

try:
    mri_classifier = load_model("Models/Is_MRI_Classifier.h5")
    mri_pred = mri_classifier.predict(image)
    mri_class = np.argmax(mri_pred, axis=1)[0]
except Exception as e:
    print("Error loading or using MRI classifier:", str(e))
    mri_class = -1

if mri_class != 1:
    print("⚠️ The uploaded image is not recognized as a brain MRI. Aborting prediction.")
else:
    print("✅ Image verified as brain MRI. Proceeding with tumor prediction...")

    # === STEP 2: Load Tumor Classifier and Predict ===
    model = load_model("Models/Brain_cancer_model.h5")
    print("\nModel loaded successfully.")

model = load_model("Models/Brain_cancer_model.h5")

# Make predictions
print("Predicting image...")
pred = model.predict(image)
pred = np.argmax(pred, axis=1)[0]


# Map prediction to class category
if pred == 0:
    pred = "Glioma Tumor"
elif pred == 1:
    print("The model predicts that there is no tumor in image.")
elif pred == 2:
    pred = "Meningioma Tumor"
else:
    pred = "Pituitary Tumor"
    
# Display prediction
if pred != 1:
    print(f"The Model predicts that it is an image with {pred}.")