import numpy as np
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)
print("Training Reviews:", len(x_train))
print("Testing Reviews:", len(x_test))

sample_review = x_test[0]

x_train = keras.utils.pad_sequences(x_train, maxlen=250)
x_test = keras.utils.pad_sequences(x_test, maxlen=250)
print("Training Shape:", x_train.shape)
print("Testing Shape:", x_test.shape)

model = keras.Sequential([
    keras.layers.Input(shape=(250,)),
    keras.layers.Embedding(10000, 32),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2
)

pred = model.predict(x_test[:5], verbose=0)
pred_class = (
    pred >= 0.5
).astype(int)
print("\nPredicted:", pred_class.flatten())
print("Actual   :", y_test[:5])

word_index = keras.datasets.imdb.get_word_index()

reverse_index = {
    value + 3: key
    for key, value in word_index.items()
}

sample_review_text = " ".join(
    reverse_index.get(i, "?")
    for i in sample_review
)

prediction = model.predict(
    x_test[:1],
    verbose=0
)
print("Sample Review:")
print(sample_review_text)

print(
    "Actual Sentiment:",
    "Positive" if y_test[0] == 1 else "Negative"
)

print(
    "Predicted Sentiment:",
    "Positive" if prediction >= 0.5 else "Negative"
)

loss, acc = model.evaluate(
    x_test,
    y_test,
    verbose=0
)
print("\nTest Accuracy:", acc)