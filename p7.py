import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
import kagglehub, os

path = kagglehub.dataset_download("tongpython/cat-and-dog")

train_dir = os.path.join(path, "training_set/training_set")
test_dir = os.path.join(path, "test_set/test_set")

train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=32
)
test_ds = keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(224, 224),
    batch_size=32
)

base = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base.trainable = False

model = keras.Sequential([
    base,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(2, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    epochs=5
)

loss, acc = model.evaluate(test_ds)
print("Test Accuracy:", acc)
