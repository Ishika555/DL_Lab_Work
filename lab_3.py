# Lab 3: CIFAR-10 Image Classification using CNN
# Name: Ishika Sharma
# Roll No: 202401100300129
# Description: This program implements a CNN model on CIFAR-10 dataset


# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# STEP 2: LOAD CIFAR-10 DATASET
# ============================================================
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()


# ============================================================
# STEP 3: NORMALIZE IMAGES (0–255 → 0–1)
# ============================================================
x_train = x_train / 255.0
x_test = x_test / 255.0


# ============================================================
# STEP 4: VISUALIZE SAMPLE IMAGES
# ============================================================
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

plt.figure(figsize=(6, 6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis('off')
plt.show()


# ============================================================
# STEP 5: APPLY DATA AUGMENTATION
# ============================================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])


# ============================================================
# STEP 6: BUILD CNN ARCHITECTURE
# ============================================================
model = models.Sequential([
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# ============================================================
# STEP 7: COMPILE THE MODEL
# ============================================================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ============================================================
# STEP 8: TRAIN THE MODEL (FORWARD + BACKPROPAGATION)
# ============================================================
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)


# ============================================================
# STEP 9: EVALUATE MODEL ON TEST DATA
# ============================================================
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_accuracy)


# ============================================================
# STEP 10: MAKE PREDICTIONS
# ============================================================
predictions = model.predict(x_test)


# ============================================================
# STEP 11: PLOT ACCURACY GRAPH
# ============================================================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Graph')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Graph')

plt.tight_layout()
plt.show()


# ============================================================
# STEP 12: END
# ============================================================
