# Cell 1: Load .h5 or .keras model + test generator + predictions (run only once)
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

# UPDATE THIS PATH TO YOUR UPLOADED MODEL
model_path = '/kaggle/input/custom-cnn-test/keras/default/1/custom_cnn_waste_classifier.h5'  # or .h5
model = load_model(model_path)

# Class names
class_names = ['Organic', 'Recyclable']

# --------------------- LIVE PREDICTION FUNCTION ---------------------
def predict_waste_image(image_path):
    # Load and preprocess image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    
    # Predict
    prediction = model.predict(img_array)[0][0]
    
    # Determine class and confidence
    if prediction < 0.5:
        predicted_class = class_names[0]
        confidence = 1 - prediction
    else:
        predicted_class = class_names[1]
        confidence = prediction
    
    # Display result
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f'Prediction: {predicted_class}\nConfidence: {confidence:.4f}', 
              fontsize=18, color='green' if confidence > 0.8 else 'red')
    plt.axis('off')
    plt.show()
    
    print(f"Predicted: {predicted_class} | Confidence: {confidence:.4f}")
    
    return predicted_class, confidence

# --------------------- TEST ON YOUR IMAGES ---------------------
# Example: replace with your uploaded image paths
test_image_paths = [
    '/kaggle/input/waste-classification-data/DATASET/TEST/O/O_12823.jpg'
]

# Run predictions
for path in test_image_paths:
    if os.path.exists(path):
        print(f"\nTesting: {os.path.basename(path)}")
        predict_waste_image(path)
    else:
        print(f"Image not found: {path}")
