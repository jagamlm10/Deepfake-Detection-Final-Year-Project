# Importing Modules 
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

# Directories
dataset_dir = './new_dataset'
train_dir = 'Train'
test_dir = 'Test'
val_dir = 'Validation'

# Hyperparameters
learning_rate = 0.0001 # Original : 0.0001
epochs = 20 # Original : 10
batch_size = 32  
image_size = (64,64) # Original : (64,64)

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

# Load datasets
train_data = get_image_dataset_from_directory(train_dir)
val_data = get_image_dataset_from_directory(val_dir)
test_data = get_image_dataset_from_directory(test_dir)

# Build the model
model = models.Sequential()
model.add(layers.Input(shape=(image_size[0], image_size[1], 3)))
model.add(layers.Rescaling(1./255, name='rescaling')) # Original : 1./127

# Convolutional Layers
model.add(layers.Conv2D(16, (3, 3), strides=1, padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))

model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))

# Flatten and Dense Layers
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))  
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu')) 
model.add(layers.Dropout(0.4)) # Original : 0.5
model.add(layers.Dense(64, activation='relu'))   
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy', 
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
)

# Callbacks
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-7, verbose=1)
model_checkpoint_callback = ModelCheckpoint('./models/deepfake_detector_model_best_2_convo_layers.keras', monitor='val_loss', save_best_only=True, verbose=1)

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[early_stopping_callback, reduce_lr_callback, model_checkpoint_callback]
)

# Evaluate the model
evaluation_metrics = model.evaluate(test_data)
print('Loss :', evaluation_metrics[0] * 100, "%")
print('Accuracy :', evaluation_metrics[1] * 100, "%")
print('Precision :', evaluation_metrics[2] * 100, "%")
print('Recall :', evaluation_metrics[3] * 100, "%")


# Plot Training Loss vs. Validation Loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot Training Accuracy vs. Validation Accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Saving the Model
model.save('./results/non-facial/deepfake_detector_model_2_convo_layers.keras')
