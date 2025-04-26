import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy

m, n, N = 1000, 15, 3

X = np.random.rand(m, n)
Y = np.zeros((m, N), dtype = int)
indices = np.random.randint(0, N, size = m)
Y[np.arange(m), indices] = 1

model = Sequential([
    Dense(units = 15, activation = "relu"),
    Dense(units = 10, activation = "relu"),
    Dense(units = 5, activation = "relu"),
    Dense(units = 3, activation = "linear")
])

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 10e-3),    # Adam optimization Algorithm
    loss = CategoricalCrossentropy(from_logits = True),
    metrics = ['categorical_accuracy']
)

model.fit(X, Y, epochs = 1000)
logits = model(X)
Y_hat = tf.nn.softmax(logits)
predicted_category_indices = np.argmax(Y_hat, axis = 1)
Y_hat = tf.one_hot(predicted_category_indices, depth = 3)
print(Y_hat)