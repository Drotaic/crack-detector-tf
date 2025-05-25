import tensorflow as tf
import numpy as np
import cv2
import sys
import os

# Load the model
MODEL_PATH = '../models/saved_model'
model = tf.keras.models.load_model(MODEL_PATH)

# Define image preprocessing
def preprocess_image(image_path, img_size=224):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Predict function
def predict(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)[0][0]
    label = "Crack" if prediction > 0.5 else "No Crack"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    print(f"Prediction: {label} (Confidence: {confidence:.2f})")

# Run from command line
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        predict(sys.argv[1])
