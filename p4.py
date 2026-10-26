import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
print("Training shape:", x_train.shape)
print("Testing shape :", x_test.shape)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape((len(x_train), 28 * 28))
x_test = x_test.reshape((len(x_test), 28 * 28))
print("Training shape:", x_train.shape)
print("Testing shape:", x_test.shape)

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(784, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

model.fit(
    x_train, x_train,
    epochs=5,
    batch_size=256,
    validation_data=(x_test, x_test)
)

reconstructed = model.predict(
    x_test[:5],
    verbose=0
)

plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(
        x_test[i].reshape(28, 28),
        cmap="gray"
    )
    plt.axis("off")

    plt.subplot(2, 5, i + 6)
    plt.imshow(
        reconstructed[i].reshape(28, 28),
        cmap="gray"
    )
    plt.axis("off")
plt.show()

original = 784
compressed = 32
print("Original features    :", original)
print("Compressed features  :", compressed)
print("Compression ratio   :", original / compressed)