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
# from IPython.display import display,clear_output


    
X_train = []
y_train = []
labels = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']


# Load and append dataset for preprocessing
for i in labels:
    train_path = "Dataset/Training/" + i
    for j in tqdm(os.listdir(train_path)):
        img = cv.imread(train_path + "/" + j)
        
        # Display image
        cv.imshow(img)
        cv.waitkey(0)
        
        # Resize
        img = cv.resize(img, (150, 150))
        X_train.append(img)
        y_train.append(i)
        print("Appended")
