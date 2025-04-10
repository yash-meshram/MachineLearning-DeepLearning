import numpy as np

def sigmoid(z):
    return 1/(1 + np.exp(-z))

x = np.array([[200, 17]])              #1x2

def dense(a_in, W, b):
    # units = number of neurons
    units = W.shape[0]
    z = np.dot(W, a_in) + b
    return sigmoid(z)

def sequential(x):
    a1 = dense(np.transpose(x), W1, b1)
    a2 = dense(a1, W2, b2)
    y_hat = 1 if a2[0] >= 0 else 0
    return y_hat, a2, a1

W1 = np.array([
    [1, 2],
    [-3, 4],
    [5, 6]
])              #3x2
b1 = np.array([
    [-1],
    [1],
    [2]
])              #3x1

W2 = np.array([[-7, 8, 9]])     #1x3
b2 = np.array([3])              #1x1

y_hat, a2, a1 = sequential(x)

print(f"x shape = {x.shape}")
print(f"W1 shape = {W1.shape}")
print(f"a1 shape = {a1.shape}\n")

print(f"W2 shape = {W2.shape}")
print(f"a2 shape = {a2.shape}\n\n")

print(f"x = {x}")
print(f"a1 = {a1}")
print(f"a2 = {a2}")
print(f"y_hat = {y_hat}")
