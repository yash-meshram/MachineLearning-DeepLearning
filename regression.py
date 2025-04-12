import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    """
    Compute the sigmoid function.

    Parameters:
    z (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Sigmoid of the input array.
    """
    return 1 / (1 + np.exp(-z))


# fun = np.dot(w, x) + b            # Linear Regression
# fun = sigmoid(np.dot(x, w) + b)   # Logistic Regression


def linear_f(x, w, b):
    """
    Linear function for regression.

    Parameters:
    x (numpy.ndarray): The input data points.
    y (numpy.ndarray): The target values.
    w (numpy.ndarray): The weights.
    b (float): The bias.

    Returns:
    numpy.ndarray: The output of the linear function.
    """
    return np.dot(x, w) + b


def logistic_f(x, w, b):
    """
    Logistic function for regression.

    Parameters:
    x (numpy.ndarray): The input data points.
    y (numpy.ndarray): The target values.
    w (numpy.ndarray): The weights.
    b (float): The bias.

    Returns:
    numpy.ndarray: The output of the logistic function.
    """
    return sigmoid(np.dot(x, w) + b)


def cost_function(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    lambda_: float,
    reg_type="linear",
):
    """
    Calculate the cost function for a given set of parameters and data points.

    Parameters:
    x (numpy.ndarray): The input data points.
    y (numpy.ndarray): The target values.
    fun (function): The function to evaluate the cost.

    Returns:
    float: The calculated cost.
    """
    # Ensure x, y and w are numpy arrays
    x = np.array(x)
    y = np.array(y)
    w = np.array(w)

    m, _ = x.shape  # Number of samples and features

    linear_f_wb_x = lambda x: np.dot(w, x) + b  # Linear function  # noqa: E731
    logistic_f_wb_x = lambda x: sigmoid(
        np.dot(x, w) + b
    )  # Logistic function  # noqa: E731

    # Calculate the cost function
    if reg_type == "linear":
        cost = (1 / (2 * m)) * np.sum((linear_f_wb_x(x) - y) ** 2) + (
            lambda_ / (2 * m)
        ) * np.sum(w**2)
    elif reg_type == "logistic":
        cost = (-1 / m) * np.sum(
            y * np.log(logistic_f_wb_x(x)) + (1 - y) * np.log(1 - logistic_f_wb_x(x))
        ) + (lambda_ / (2 * m)) * np.sum(w**2)

    return cost


def gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    alpha: float,
    lambda_: float,
    iteration: int = 1000,
    reg_type="linear",
):
    """
    Perform gradient descent to optimize the parameters.

    Parameters:
    x (numpy.ndarray): The input data points.
    y (numpy.ndarray): The target values.
    w (numpy.ndarray): The weights.
    b (float): The bias.
    alpha (float): The learning rate.
    lambda_ (float): The regularization parameter.

    Returns:
    tuple: Updated weights and bias.
    """
    m, _ = x.shape  # Number of samples and features

    cost_array = []  # Initialize an empty array to store cost values

    for i in range(iteration):
        linear_f_wb_x = lambda x: np.dot(w, x) + b  # Linear function  # noqa: E731
        logistic_f_wb_x = lambda x: sigmoid(
            np.dot(x, w) + b
        )  # Logistic function  # noqa: E731

        if reg_type == "linear":
            dw = (1 / m) * np.dot(x.T, (linear_f_wb_x(x) - y)) + (lambda_ / m) * w
            db = (1 / m) * np.sum(linear_f_wb_x(x) - y)
        elif reg_type == "logistic":
            dw = (1 / m) * np.dot(x.T, (logistic_f_wb_x(x) - y)) + (lambda_ / m) * w
            db = (1 / m) * np.sum(logistic_f_wb_x(x) - y)

        w -= alpha * dw
        b -= alpha * db

        current_cost = cost_function(x, y, w, b, lambda_, reg_type)
        cost_array.append(current_cost)  # Store the current cost value

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {current_cost}, w = {w}, b = {b}")

    plt.plot(cost_array, label="Cost")

    return w, b


x = np.array(
    [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ]
)  # 5 samples, 5 features

y = np.array([10, 20, 30, 40, 50])  # 5 target values

w = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # 5 weights (one for each feature)
b = 0.5  # Bias
lambda_ = 0.01

cost = cost_function(x, y, w, b, lambda_)
print("Cost:", cost)

w_new, b_new = gradient_descent(x, y, w, b, 0.001, lambda_)
print("Updated weights:", w_new)
print("Updated bias:", b_new)

cost_new = cost_function(x, y, w_new, b_new, lambda_)
print("New Cost:", cost_new)

x_pred = np.array(
    [
        [26, 27, 28, 29, 30],
        [31, 32, 33, 34, 35],
        [36, 37, 38, 39, 40],
        [81, 82, 83, 84, 85],
    ]
)  # New input data points
y_pred = linear_f(x_pred, w_new, b_new)  # Predict using the updated weights and bias
print("Predicted values:", y_pred)
