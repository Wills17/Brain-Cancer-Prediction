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
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")
#import ipywidgets as widgets
#from IPython.display import display,clear_output



# List for arrays
X_train = []
y_train = []
folders = ["glioma_tumor", "no_tumor", "meningioma_tumor", "pituitary_tumor"]


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
    print("Appended all images in {} folder successfully.".format(folder))
    
print(f"\nAppended all images in the Training folder successfully.")
        
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
    print("Appended all images {} folder successfully".format(folder))
    
print(f"\nAppended all images in Testing folder successfully.")
        
        
# Convert all images to numpy arrays
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
    # cv.imshow(f"Random Image from "{folder}" folder", cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # cv.waitKey(10000)
    # cv.destroyAllWindows()



# Split into test and train sets
X_train,X_test,y_train,y_test = train_test_split(X_train,y_train, test_size=0.2, random_state=32)
print("\nShape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)


# Convert y_train into categorical (numerical) value
new_y_train = []
for folder in y_train:
    new_y_train.append(folders.index(folder))
y_train = new_y_train
y_train = tf.keras.utils.to_categorical(y_train)
# print("y_train:", y_train)


new_y_test = []
for i in y_test:
    new_y_test.append(folders.index(folder))
y_test = new_y_test
y_test = tf.keras.utils.to_categorical(y_test)
# print("y_test:", y_test)



# Initialize EfficientNetB0 model with pre-trained ImageNet weights, excluding the top classification layer, 
# and sets the input shape to (150, 150, 3). 
# Leverage transfer learning for feature extraction dataset.
effnet = EfficientNetB0(weights="imagenet",include_top=False,input_shape=(150,150,3))
# Save the effnet model to device
effnet.save("EfficientNetB0_model.h5")
print("\nEfficientNetB0 model saved as 'EfficientNetB0_model.h5' successfully.")

model = effnet.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)
model = tf.keras.layers.Dropout(rate=0.5)(model)
model = tf.keras.layers.Dense(4,activation="softmax")(model)
model = tf.keras.models.Model(inputs=effnet.input, outputs = model)

print(model.summary())

# Compile model
print("\nCompiling model...")
model.compile(loss="categorical_crossentropy",optimizer = "Adam", metrics= ["accuracy"])

# Set up callbacks for training: TensorBoard for logging, 
# ModelCheckpoint to save the best model, 
# and ReduceLROnPlateau to adjust learning rate on plateau.
tensorboard = TensorBoard(log_dir = "logs")
checkpoint = ModelCheckpoint("Brain_cancer_model.h5", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
reduce_lr = ReduceLROnPlateau(monitor = "val_accuracy", factor = 0.3, patience = 2, min_delta = 0.001,
                              mode="auto",verbose=1)


# Model training 
print("\nTraining model...")
model= model.fit(X_train,y_train,validation_split=0.1, epochs=15, verbose=1, batch_size=32,
                   callbacks=[tensorboard,checkpoint,reduce_lr])


# Plot training & validation accuracy and loss values
# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(model.history["accuracy"], label="Train Accuracy")
plt.plot(model.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(model.history["loss"], label="Train Loss")
plt.plot(model.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()


# Print message showing model has been saved
print("\nModel saved as 'Brain_cancer_model.h5' successfully.")


# Make predictions
print("Test predictions...")
prediction = model.predict(X_test)
prediction = np.argmax(prediction, axis=1)
new_y_test = np.argmax(y_test, axis=1)

# Evaluate model
print("Classification Report:", classification_report(new_y_test,prediction))

