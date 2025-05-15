# Import libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import cv2 as cv
import io
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report,confusion_matrix
#import ipywidgets as widgets
#from IPython.display import display,clear_output



# List for arrays
X_train = []
y_train = []
folders = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']


# Load and append dataset for preprocessing
# Train folder
for folder in folders:
    train_path = "Dataset/Training/" + folder
    print("\nCurrent folder:", folder)
    for x in tqdm(os.listdir(train_path)):
        img = cv.imread(train_path + "/" + x)
        
        # Display image
        # cv.imshow(f"Original Image {j}", img)
        # cv.waitKey(1000)
        
        # Resize and display resized
        img = cv.resize(img, (150, 150))
        # cv.imshow(f"Resized Image {j}", img)
        # cv.waitKey(1000)
        # cv.destroyAllWindows()
        
        X_train.append(img)
        y_train.append(folder)
    print("Appended all images in folder {} sucessfully.".format(folder))
    
print(f"\nAppended all images in the Training folder sucessfully.")
        
#Test folder
for folder in folders:
    train_path = "Dataset/Testing/" + folder
    print("\nCurrent folder:", folder)
    for y in tqdm(os.listdir(train_path)):
        img = cv.imread(train_path + "/" + y)
        
        # Display image
        # cv.imshow(f"Original Image {j}", img)
        # cv.waitKey(1000)
        
        # Resize and display resized
        img = cv.resize(img, (150, 150))
        # cv.imshow(f"Resized Image {j}", img)
        # cv.waitKey(1000)
        # cv.destroyAllWindows()
        
        X_train.append(img)
        y_train.append(folder)
    print("Appended all images folder {} sucessfully".format(folder))
print(f"\nAppended all images in Testing folder sucessfully.")
        
        
# Conert all images to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
print("\nShape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)



# Display a random image from each folder
for folder in folders:
    folder_path = "Dataset/Training/" + folder
    random_image = np.random.choice(os.listdir(folder_path))
    img = cv.imread(folder_path + "/" + random_image)
    img = cv.resize(img, (500, 500))
    # cv.imshow(f"Random Image from '{folder}' folder", cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # cv.waitKey(10000)
    # cv.destroyAllWindows()


# Split into test and train sets
X_train,X_test,y_train,y_test = train_test_split(X_train,y_train, test_size=0.2,random_state=34)
print("\nShape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
