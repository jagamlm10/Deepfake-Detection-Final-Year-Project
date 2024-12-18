import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

model_path = './results/non-facial/deepfake_detector_model_non_facial.keras'

model = load_model(model_path)

# Directories
dataset_dir = './non_facial_dataset'
train_dir = 'Train'
test_dir = 'Test'
val_dir = 'Validation'

batch_size = 32 
image_size = (32,32)

# Load Image Dataset Function
def get_image_dataset_from_directory(dir_name):
    dir_path = os.path.join(dataset_dir, dir_name)
    return tf.keras.preprocessing.image_dataset_from_directory(
        dir_path,
        labels='inferred',
        label_mode='binary',
        color_mode='rgb',
        seed=42,
        batch_size=batch_size,
        image_size=image_size
    )

# Load dataset
test_data = get_image_dataset_from_directory(test_dir)

# Get predictions for confusion matrix
y_true = []
y_pred = []

for images, labels in test_data:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend((preds > 0.5).astype(int).flatten()) 

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
disp.plot(cmap=plt.cm.Greens)
plt.title('Confusion Matrix')
plt.show()