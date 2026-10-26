%pip install -q yfinance
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

data = yf.download(
    'AAPL',
    start='2023-01-01',
    end='2024-01-01'
)
prices = data['Close'].values.reshape(-1, 1)
print(data.head())

def create_data(s, steps=20):
    X = np.array([
        s[i:i+steps, 0]
        for i in range(len(s) - steps)
    ])
    y = np.array([
        s[i+steps, 0]
        for i in range(len(s) - steps)
    ])
    return X, y

scaler = MinMaxScaler()
scaled = scaler.fit_transform(prices)

steps = 20
X, y = create_data(scaled, steps)
X = X.reshape((-1, steps, 1))
print("X shape:", X.shape)
print("y shape:", y.shape)

model = Sequential([
    LSTM(32, input_shape=(steps, 1)),
    Dense(1)
])
model.summary()

model.compile(
    optimizer='adam',
    loss='mse'
)

model.fit(
    X,
    y,
    epochs=15,
    batch_size=8,
    verbose=0
)

pred_scaled = model.predict(X)
pred = scaler.inverse_transform(pred_scaled)

plt.plot(
    prices[steps:],
    label='True'
)
plt.plot(
    pred,
    label='Predicted'
)
plt.legend()
plt.show()