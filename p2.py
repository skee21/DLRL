%pip install -q tensorflow matplotlib seaborn scikit-learn pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import confusion_matrix, f1_score

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)

sample_no = 1050
plt.imshow(x_train[sample_no].reshape(28, 28), cmap="gray")
plt.title(f"Ground Truth: {y_train[sample_no]}")
plt.axis("off")
plt.show()
print("Sample number:", sample_no)
print("Ground truth:", y_train[sample_no])

x_train = (x_train.astype("float32") / 255.0).reshape(-1, 784)
x_test = (x_test.astype("float32") / 255.0).reshape(-1, 784)
print("Flattened training shape:", x_train.shape)
print("Flattened testing shape:", x_test.shape)

model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train,
    y_train,
    epochs=4,
    batch_size=32,
    validation_split=0.1
)

history_df = pd.DataFrame(history.history)
history_df[["loss", "val_loss"]].plot()
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

history_df[["accuracy", "val_accuracy"]].plot()
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

predictions = model.predict(x_test[:5], verbose=0)
predicted_classes = np.argmax(predictions, axis=1)
print("Predicted classes:", predicted_classes)
print("Actual classes   :", y_test[:5])

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)
print(f"Test loss     : {test_loss:.4f}")
print(f"Test accuracy : {test_accuracy:.4f}")

all_predictions = model.predict(x_test, verbose=0)
all_classes = np.argmax(all_predictions, axis=1)

cm = confusion_matrix(y_test, all_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix")
plt.show()

f1 = f1_score(
    y_test,
    all_classes,
    average="weighted"
)
print(f"F1 Score: {f1:.4f}")
print(f"F1 Score: {f1:.4f}")