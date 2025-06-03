import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --------------------
# CONFIGURATION
# --------------------
MODEL_PATH = 'fine_tuned_model.h5'
IMAGE_SIZE = (224, 224)

# --------------------
# LOAD THE MODEL
# --------------------
print("Loading the model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# --------------------
# LOAD AND PREPROCESS IMAGE
# --------------------
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # normalize
    return img_array

# --------------------
# PREDICT FUNCTION
# --------------------
def predict_image(img_path):
    img_array = load_and_preprocess_image(img_path)
    prediction = model.predict(img_array)[0][0]  # single output, sigmoid
    return prediction

# --------------------
# MAIN
# --------------------
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    prediction = predict_image(image_path)

    # Threshold for classification (0.5 by default)
    if prediction >= 0.5:
        print(f"Prediction: CRACK DETECTED ({prediction:.2f})")
    else:
        print(f"Prediction: NO CRACK ({prediction:.2f})")
