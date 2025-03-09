from fastapi import FastAPI, UploadFile, File
from tensorflow import keras
import numpy as np
import uvicorn
import cv2

app = FastAPI()

# Load the model
model = keras.models.load_model("mnist_model.h5")

def preprocess_image(image_bytes):
    image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    image = cv2.resize(image, (28, 28))  # Resize to 28x28
    image = image / 255.0  # Normalize
    image = image.reshape(1, 28, 28)  # Reshape for model input
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)
    prediction = model.predict(image)
    predicted_digit = int(np.argmax(prediction))
    return {"predicted_digit": predicted_digit}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)