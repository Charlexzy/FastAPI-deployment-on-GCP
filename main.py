from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf

# Define the same data augmentation parameters as during training
data_augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

app = FastAPI()

MODEL_PATH = "gs://potatodiseaseclassifier/models/potatoes.h5"

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Load the model
MODEL = None

def load_model():
    global MODEL
    if MODEL is None:
        MODEL = tf.keras.models.load_model(MODEL_PATH)

# Preprocess the image
def preprocess_image(image):
    # Resize the image to the desired dimensions
    image = image.resize((256, 256))
    # Convert the image to an array
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    # Expand the dimensions of the image to match the expected input shape
    img_array = np.expand_dims(img_array, axis=0)
    # Apply data normalization
    img_array = img_array / 255.0
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load the model if not loaded already
    load_model()

    # Read the uploaded file as an image
    image = Image.open(BytesIO(await file.read()))

    # Preprocess the image
    img_array = preprocess_image(image)

    # Make predictions by calling the loaded model on the preprocessed image
    predictions = MODEL.predict(img_array)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {
        'class': predicted_class,
        'confidence': confidence
    }
    