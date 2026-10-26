%pip install -q tensorflow matplotlib seaborn scikit-learn pandas kagglehub
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
import kagglehub
from PIL import Image
from keras.preprocessing.image import img_to_array
from sklearn.metrics import confusion_matrix

path = kagglehub.dataset_download(
    "tongpython/cat-and-dog"
)
train_dir = os.path.join(path, "training_set", "training_set")
test_dir = os.path.join(path, "test_set", "test_set")
print("Dataset path:", path)

train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(128, 128),
    batch_size=32
)

test_ds = keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(128, 128),
    batch_size=32
)

print("Classes:", train_ds.class_names)

train_count = len(train_ds.file_paths)
test_count = len(test_ds.file_paths)
print("Training images:", train_count)
print("Testing images:", test_count)
print("Total images   :", train_count + test_count)

model = keras.Sequential([
    keras.layers.Rescaling(
        1./255,
        input_shape=(128, 128, 3)
    ),
    keras.layers.Conv2D(
        32, 3, activation="relu"
    ),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(
        64, 3, activation="relu"
    ),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(
        64, activation="relu"
    ),
    keras.layers.Dense(
        1, activation="sigmoid"
    )
])

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=5
)

loss, accuracy = model.evaluate(
    test_ds
)
print("Test Accuracy:", accuracy)

for images, labels in test_ds.take(1):
    prediction = model.predict(
        images, verbose=0
    )[0][0]
    plt.imshow(
        images[0].numpy().astype("uint8")
    )
    plt.axis("off")
    plt.show()

print(
    "Actual:",
    "Dog" if labels[0] == 1 else "Cat"
)
print(
    "Predicted:",
    "Dog" if prediction >= 0.5 else "Cat"
)
)