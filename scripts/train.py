import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, GRU
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --------------------
# CONFIGURATION
# --------------------
PRETRAINED_MODEL_PATH = 'pretrained_model.h5'  # Update this to your path
TRAIN_DIR = 'data/train'                       # Your training images
VAL_DIR = 'data/validation'                    # Your validation images
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

# --------------------
# LOAD PRETRAINED MODEL
# --------------------
print("Loading pre-trained model...")
base_model = load_model(PRETRAINED_MODEL_PATH)
base_model.trainable = False  # Freeze base model layers

# --------------------
# ADD CUSTOM LAYERS
# --------------------
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

# Optional: Add RNNs (example with LSTM)
# x = tf.expand_dims(x, axis=1)  # Add sequence dimension if needed
# x = LSTM(64, return_sequences=False)(x)

outputs = Dense(1, activation='sigmoid')(x)  # Binary classification

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --------------------
# DATA PREPARATION
# --------------------
print("Preparing data generators...")
train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   rotation_range=20,
                                   zoom_range=0.2)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary')

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary')

# --------------------
# TRAIN THE MODEL
# --------------------
print("Starting training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# --------------------
# SAVE THE FINE-TUNED MODEL
# --------------------
print("Saving fine-tuned model...")
model.save('fine_tuned_model.h5')
print("Model saved as fine_tuned_model.h5")
