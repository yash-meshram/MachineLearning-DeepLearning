import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy 

m, n, N = 100, 5, 3

X = np.random.rand(m, n)
Y = np.zeros((m, N), dtype = int)
indices = np.random.randint(0, N, size = m)
Y[np.arange(m), indices] = 1

model = Sequential([
    Dense(units = 25, activation = 'relu'),
    Dense(units = 15, activation = 'relu'),
    Dense(units = 3, activation = 'softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss=CategoricalCrossentropy(),  # change here
    metrics=['categorical_accuracy']
)

model.fit(X, Y, epochs = 100)

y_hat = model.predict(X)
predicted_category_index = np.argmax(y_hat, axis = 1)
y_hat = np.zeros_like(y_hat)
y_hat[np.arange(len(y_hat)), predicted_category_index] = 1
print(f"y_hat = {y_hat}")

# Handelling Numerical roundoff error - More Numerical accurate
model_new = Sequential([
    Dense(units = 25, activation = 'relu'),
    Dense(units = 15, activation = 'relu'),
    Dense(units = N, activation = 'linear')
])
model_new.compile(
    optimizer=tf.keras.optimizers.RMSprop(),
    loss=CategoricalCrossentropy(from_logits = True),  # change here
    metrics=['categorical_accuracy']
)
model_new.fit(X, Y, epochs = 1000)
logits = model_new(X)               # tensor dtype
y_hat_new = tf.nn.softmax(logits)
predicted_category_index = tf.argmax(y_hat_new, axis=1)     #"For each row, find the index of the max value across the columns."
y_hat_new = tf.one_hot(predicted_category_index, depth=3)   #"I want the one-hot vectors to have 3 elements, because I have 3 possible categories (classes)."
print(f"y_hat_new = {y_hat_new}")
