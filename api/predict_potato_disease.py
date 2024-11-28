import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sys
model_path = 'C:/Users/vasan/OneDrive/Desktop/Deep-learning/potato_disease/saved_models/1.keras'
model = load_model(model_path)

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))  # Change target_size based on your model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image to match the training process
    return img_array

# Function to make a prediction
def predict_disease(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    return prediction

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
    else:
        img_path = sys.argv[1]
        
        prediction = predict_disease(img_path)
        print("Prediction:", prediction)