import numpy as np

def sigmoid(z):
    return 1/(1 + np.exp(-z))

X = np.array([[200, 17]])       #1x2

W = np.array([
    [1, -2],
    [-3, 4],
    [5, -6]
])                              #3x2

B = np.array([[-1, 1, 2]])      #1x3

def dense(A_in, W, B):
    z = np.matmul(A_in, W.T) + B
    return sigmoid(z)

A1 = dense(X, W, B)

print(A1)
print(A1.shape)
