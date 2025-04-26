import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Loading the data
df = pd.read_csv('data/model_evaluation_and_selection_dataset.csv', header = None)

# splitting the input and target
x = df.iloc[:,0]
y = df.iloc[:,1]

# expanding the dimension of x and y from 1D --> 2D
x = np.expand_dims(x, axis = 1)
y = np.expand_dims(y, axis = 1)

print(f"Shape of input x = {x.shape}")
print(f"Shape of output y = {y.shape}")

# plot the dataset
plt.scatter(x, y)
plt.xlabel('input x')
plt.ylabel('target y')
plt.title("input vs target")
plt.show()

# splitting the dataset
# df ---> df_train (60%) + df_cv (20%) + df_test (20%)
x_train, x_, y_train, y_ = train_test_split(
    x, y, train_size = 0.60, random_state = 1
)
x_cv, x_test, y_cv, y_test = train_test_split(
    x_, y_, train_size = 0.50, random_state = 1
)
del x_, y_
print(f"Shape of the training set = {x_train.shape}, {y_train.shape}")
print(f"Shape of the cross validation set = {x_cv.shape}, {y_cv.shape}")
print(f"Shape of the testing set = {x_test.shape}, {y_test.shape}")

# plotting training, cross validation and testing set
plt.scatter(x_train, y_train, label = 'training')
plt.scatter(x_cv, y_cv, marker = 'x', label = 'cross-validation')
plt.scatter(x_test, y_test, marker = 's', label = 'testing')
plt.legend()
plt.xlabel('input x')
plt.ylabel('target y')
plt.title("input vs target")
plt.show()

# Scaling
scaler = StandardScaler()       # z-score scaling
x_train_scaled = scaler.fit_transform(x_train)
x_cv_scaled = scaler.transform(x_cv)
x_test_scaled = scaler.transform(x_test)

# Single model only - Linear regression
# --------------------------------------
# building the model
model = LinearRegression()
# training the model
model.fit(x_train_scaled, y_train)
# evaluating the model
y_train_hat = model.predict(x_train_scaled)
y_cv_hat = model.predict(x_cv_scaled)
print(f"Training MSE = {mean_squared_error(y_train, y_train_hat) / 2}")
print(f"Cross-Validation MSE = {mean_squared_error(y_cv, y_cv_hat) / 2}")

# Single model only - Polynomial regression (dregree = 2)
# -------------------------------------------------------

# adding x**2 in x_train, x_cv and x_test
# data --> Scale it first, then --> apply PolynomialFeatures to generate polynomial features
poly = PolynomialFeatures(degree = 2, include_bias = False)

# in fit_tranform we will use the already scale data (x_train_scaled) and not un-scaled data (x_train)
x_train_scaled_poly = poly.fit_transform(x_train_scaled)
x_cv_scaled_poly = poly.transform(x_cv_scaled)

# training the model
model.fit(x_train_scaled_poly, y_train)

# evaluating teh model
y_train_hat = model.predict(x_train_scaled_poly)
y_cv_hat = model.predict(x_cv_scaled_poly)
print(f"Training MSE = {mean_squared_error(y_train, y_train_hat) / 2}")
print(f"Cross-Validation MSE = {mean_squared_error(y_cv, y_cv_hat) / 2}")

# plotting x**2, after Scale --> Poly
plt.scatter(x_train_scaled_poly[:,1], y_train)
plt.xlabel('x_train**2')
plt.ylabel('y_train')
plt.title('Scale --> Poly')
plt.show()

# adding x**2 in x_train, x_cv and x_test
# data --> apply PolynomialFeatures to generate polynomial features, then ---> Scale it
x_train_poly = poly.fit_transform(x_train)
x_cv_poly = poly.transform(x_cv)
x_train_poly_scaled = scaler.fit_transform(x_train_poly)
x_cv_poly_scaled = scaler.transform(x_cv_poly)

# training the model
model.fit(x_train_poly_scaled, y_train)

# evaluating the model
y_train_hat = model.predict(x_train_poly_scaled)
y_cv_hat = model.predict(x_cv_poly_scaled)
print(f"Training MSE = {mean_squared_error(y_train, y_train_hat) / 2}")
print(f"Cross-Validation MSE = {mean_squared_error(y_cv, y_cv_hat) / 2}")

# plotting x**2, after Poly --> Scale
plt.scatter(x_train_poly_scaled[:,1], y_train)
plt.xlabel('x_train**2')
plt.ylabel('y_train')
plt.title('Poly --> Scale')
plt.show()

# plotting x**2, just x**2 after Poly only
plt.scatter(x_train_poly[:,1], y_train)
plt.xlabel('x_train**2')
plt.ylabel('y_train')
plt.title('x**2')
plt.show()


# *****************************************************************************************************************************************************
# NOTE = Date --> Poly --> Scale
# It is recommended to always
# Create features by PolynomialFeatures and then Scale it
# Poly --> Scale 
# *****************************************************************************************************************************************************
# NOTE = .fit_transform() v/s .transform() in Poly and Scale
# .fit_transform() --> 
# Fits the data (i.e., it learns whatever is needed from the data)
# AND
# Transforms the data (i.e., applies the learned rules)
# Look at x_train, figure out how to create all the polynomial combinations.
# Then create those new polynomial features for x_train.
# .transform() -->
# Only transforms the data (uses the rules learned earlier from .fit())
# No learning happens here.
# Use the same learned way of creating polynomial features (based on x_train)
# Apply it to x_cv.
# *****************************************************************************************************************************************************


# Multi model - Polynomial (degree - 1 to 10)
# ---------------------------------------------
# defining arrays - will store the each model details 
polys = []
scalers = []
models = []
train_mses = []
cv_mses = []

for degree in range(1, 11):
    # creating the features by PolynomialFeatures
    poly = PolynomialFeatures(degree = degree, include_bias = False)
    x_train_poly = poly.fit_transform(x_train)                  # Learn from x_train and create the mapped features *********************************
    polys.append(poly)
    
    # Scaling the features
    scaler = StandardScaler()
    x_train_poly_scaled = scaler.fit_transform(x_train_poly)    # Learn from x_train and create the mapped features *********************************
    scalers.append(scaler)
    
    # building the model
    model = LinearRegression()
    
    # training the model
    model.fit(x_train_poly_scaled, y_train)
    models.append(model)
    
    # evaluating the model
    # --------------------
    
    # compute Mean Squared Error (training set)
    y_hat = model.predict(x_train_poly_scaled)
    train_mse = mean_squared_error(y_train, y_hat) / 2
    train_mses.append(train_mse)
    
    # add polynomial features and scale the cross validation set
    # Only transform x_cv based on what was learned from x_train **************************************************************************************
    x_cv_poly = poly.transform(x_cv)
    x_cv_poly_scaled = scaler.transform(x_cv_poly)
    
    # compute Mean Squared Error (cross-validation set)
    y_hat = model.predict(x_cv_poly_scaled)
    cv_mse = mean_squared_error(y_cv, y_hat) / 2
    cv_mses.append(cv_mse)
    
degree = np.argmin(cv_mses) + 1

# plotting the cv MSE
plt.plot(np.arange(1, 11), cv_mses, 'o-', label='dev MSE')   # 'o-' means circle + line
plt.plot(np.arange(1, 11), train_mses, 'o-', label = 'train MSE')
plt.plot(np.full(10, degree), np.arange(0, 200, 20), '--', label = 'Min dev MSE')
plt.xlabel('degree')
plt.ylabel('MSE')
plt.title('degree vs MSE')
plt.legend()
plt.show()


print(f'degree = {degree}')
print(f'model = {models[degree-1]}')
print(f'train MSE = {train_mses[degree-1]}')
print(f'Cross-Validation MSE = {cv_mses[degree-1]}')

