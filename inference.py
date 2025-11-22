# Live Testing Demo
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os

model = load_model('saved_models/custom_cnn_best.keras')
class_names = ['Organic', 'Recyclable']

def predict_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0][0]
    label = class_names[1] if pred > 0.5 else class_names[0]
    conf = pred if pred > 0.5 else 1 - pred
    return img, label, conf

# Demo on 6 images
demo_folder = 'demo/input_images'
images = [f for f in os.listdir(demo_folder) if f.endswith(('.jpg', '.png'))]

plt.figure(figsize=(15, 10))
for i, img_file in enumerate(images[:6]):
    path = os.path.join(demo_folder, img_file)
    img, label, conf = predict_image(path)
    plt.subplot(2, 3, i+1)
    plt.imshow(img)
    color = 'green' if 'Organic' in label or 'Recyclable' in label else 'red'
    plt.title(f'{label}\nConfidence: {conf:.4f}', color=color, fontsize=14)
    plt.axis('off')

plt.suptitle('Custom CNN Live Testing Demo (92% Test Accuracy)', fontsize=18)
plt.tight_layout()
plt.savefig('demo/output_predictions.png')
plt.show()
