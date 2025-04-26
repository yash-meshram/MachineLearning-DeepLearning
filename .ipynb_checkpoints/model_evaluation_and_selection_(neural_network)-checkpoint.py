import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# suppress warnings
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

df = pd.read_csv('data/model_evaluation_and_selection_dataset.csv', header = None)

x = df.iloc[:, 0]
y = df.iloc[:, 1]

x = np.expand_dims(x, axis = 1)
y = np.expand_dims(y, axis = 1)

# splitting the data
x_train, x_, y_train, y_ = train_test_split(x, y, train_size = 0.60, random_state = 1)
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, train_size = 0.50, random_state = 1)
del x_, y_
print(f"Shape of the training set = {x_train.shape}, {y_train.shape}")
print(f"Shape of the cross validation set = {x_cv.shape}, {y_cv.shape}")
print(f"Shape of the testing set = {x_test.shape}, {y_test.shape}")

# Scaling the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train) 
x_cv_scaled = scaler.transform(x_cv)
x_test_scaled = scaler.transform(x_test)

# building the model
tf.random.set_seed(20)          # To get reproducible results, you need to set the random seeds
model_1 = Sequential(
    [
        Dense(units = 25, activation = 'relu'),
        Dense(units = 15, activation = 'relu'),
        Dense(units = 1, activation = 'linear')
    ],
    name = 'model_1'
)
model_2 = Sequential(
    [
        Dense(units = 20, activation = 'relu'),
        Dense(units = 12, activation = 'relu'),
        Dense(units = 12, activation = 'relu'),
        Dense(units = 20, activation = 'relu'),
        Dense(units = 1, activation = 'linear')
    ],
    name = 'model_2'
)
model_3 = Sequential(
    [
        Dense(units = 32, activation = 'relu'),
        Dense(units = 16, activation = 'relu'),
        Dense(units = 8, activation = 'relu'),
        Dense(units = 4, activation = 'relu'),
        Dense(units = 12, activation = 'relu'),
        Dense(units = 1, activation = 'linear')
    ],
    name = 'model_3'
)
models = [model_1, model_2, model_3]

train_mses = []
cv_mses = []

for model in models:
    # compile the model | setup the loss and optimization
    model.compile(
        loss = 'mse',
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)
    )
    
    print(f'Training {model.name}...')
    
    # train the model
    model.fit(x_train_scaled, y_train, epochs = 300, verbose = 0)
    
    print('Done\n')
    
    # evaluating the model
    # training set
    y_hat = model.predict(x_train_scaled, verbose = 0)
    train_mse = mean_squared_error(y_train, y_hat) / 2
    train_mses.append(train_mse)
    # cross-validation set
    y_hat = model.predict(x_cv_scaled, verbose = 0)
    cv_mse = mean_squared_error(y_cv, y_hat) / 2
    cv_mses.append(cv_mse)
    
print('\nResult:')
for i in range(len(models)):
    print(
        f'model = {models[i].name}\n' +
        f'training MSE = {train_mses[i]}   ' +
        f'cross-validation MSE = {cv_mses[i]}'
    )
    
# selected model
print('\nSelected Model:')
index_ = np.argmin(cv_mses)
print(
    f'model = {models[index_].name}\n' +
    f'training MSE = {train_mses[index_]}   ' +
    f'cross-validation MSE = {cv_mses[index_]}'
)
