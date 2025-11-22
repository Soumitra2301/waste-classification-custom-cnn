# Full test evaluation
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Paths
BASE_PATH = '/kaggle/input/waste-classification-data/DATASET'  # Change if local
test_dir = os.path.join(BASE_PATH, 'TEST')

# Load the trained model
model = load_model('saved_models/custom_cnn_best.keras')
print("Model loaded!")

# Test generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Predictions
y_pred = (model.predict(test_generator) > 0.5).astype(int).flatten()
y_true = test_generator.classes
class_labels = ['Organic', 'Recyclable']

# Classification Report
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix - Custom CNN (Test Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

print(f"\nTest Accuracy: {np.mean(y_pred == y_true):.4f}")
