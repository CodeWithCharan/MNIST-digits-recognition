from fastapi import FastAPI, UploadFile, File
from tensorflow import keras
import numpy as np
import cv2
import io

app = FastAPI()

# Load the model
model = keras.models.load_model("mnist_model.h5")

def preprocess_image(image_path):
    image = cv2.imread("test_digit.png", cv2.IMREAD_GRAYSCALE)  # Load in grayscale
    image = cv2.resize(image, (28, 28))  # Resize to 28x28
    image = image / 255.0  # Normalize
    image = image.reshape(1, 28, 28)  # Reshape for model input
    return image

def predict(image_path):
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    predicted_label = np.argmax(predictions)
    print(f"Predicted Digit: {predicted_label}")

if __name__ == "__main__":
    pass