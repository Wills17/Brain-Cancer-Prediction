"""For visual predictions"""

# Import neccessary libraries
import os
import random
import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model 
import warnings
warnings.filterwarnings("ignore")


# Load the model
model = load_model("Models/Brain_cancer_model.h5")
print("Loading model...")
print("\nModel loaded successfully.")

# Allowed extensions
extensions = ["jpg", "jpeg", "png"]

# Pick random folder and image in folder
image_path = "Dataset/Predict/"

def predict():
        """Function to make predictions on brain tumor images"""
        image_picked = random.choice(os.listdir(image_path))
        print("\nImage picked:", image_picked)
            
        # Check image extension
        if image_picked.split(".")[-1].lower() not in extensions:
            print(f"Image format is invalid: {image_picked.split(".")[-1]}")
            print("Supported extensions are: {}".format(extensions))

        # Update image path
        new_image_path = image_path + image_picked
        print("\nNew image path:", new_image_path)

        # Read and process image
        image = cv.imread(new_image_path)

        # Display original image
        cv.imshow(f"Original Image: {image_picked}", image)
        # cv.waitKey(5000)
                
        # Resize and display the resized image
        image = cv.resize(image, (150, 150))
        cv.imshow(f"Resized Image:{image_picked}", image)
        cv.waitKey(5000)
            
            
        # Ask user for confirmation
        user_input = input("\nIs this a brain MRI image? (yes/no): ").strip().lower()
        if user_input not in ['yes', 'y']:
            print("Prediction aborted. Please upload a valid brain MRI image.")
            exit()

        # Convert into numpy array
        image = np.array(image)

        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        image = image.reshape(1,150,150,3)

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
            
            
        # Ask user if they want to predict another image
        user_input = input("\nDo you want to predict another image? (yes/no): ").strip().lower()
        if user_input not in ['yes', 'y']:
            cv.destroyAllWindows()
            print("Exiting the prediction loop.")
            exit()
        else:
            cv.destroyAllWindows()
            print("Continuing to predict another image...")
            predict()

while True:    
    # Call the predict function 
    predict()


# End