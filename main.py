from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn  # ASGI
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# MODEL = tf.keras.models.load_model("../potatoes.h5", custom_objects={'tf': tf})
MODEL_PATH = "gs://potatodiseaseclassifier/models/potatoes.h5"
MODEL = tf.keras.models.load_model(MODEL_PATH, custom_objects={'tf': tf})

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

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

def preprocess_image(image):
    # Resize the image to the desired dimensions
    image = image.resize((256, 256))
    # Convert the image to an array
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    # Expand the dimensions of the image to match the expected input shape
    img_array = np.expand_dims(img_array, axis=0)
    # Apply data augmentation and normalization
    img_array = data_augmentation.flow(img_array, shuffle=False).next()
    return img_array

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
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

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
