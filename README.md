# Neural Network and Machine Learning Model Evaluation Framework

This project provides a comprehensive framework for building, training, and evaluating various machine learning models with a focus on neural networks. It implements multiple optimization algorithms, model evaluation techniques, and supports both regression and classification tasks.

The framework offers a modular approach to machine learning model development, featuring implementations of linear regression, logistic regression, and neural networks using both pure Python/NumPy and TensorFlow. It includes tools for model selection, hyperparameter tuning, and performance evaluation using training, cross-validation, and test datasets.

Key features include:
- Multiple neural network architectures with configurable layers and activation functions
- Support for both regression and classification tasks
- Implementation of optimization algorithms including gradient descent and Adam
- Comprehensive model evaluation and selection tools
- Data preprocessing utilities including feature scaling and polynomial feature generation
- Visualization tools for model performance analysis

## Repository Structure
```
.
├── adam_optimization_algorithm.py     # Implementation of Adam optimizer for neural networks
├── derivatives.py                     # Symbolic differentiation utilities using SymPy
├── model_evaluation_and_selection_(neural_network).py    # Neural network model evaluation framework
├── model_evaluation_and_selection.py  # General model evaluation and selection tools
├── neural_network_1.py               # Basic neural network implementation with one hidden layer
├── neural_network_2.py               # Neural network with two hidden layers
├── neural_network_3.py               # Enhanced neural network implementation
├── neural_network_tensorflow.py       # TensorFlow-based neural network implementation
├── regression.py                      # Linear and logistic regression implementations
└── utils.py                          # Utility functions for data processing and visualization
```

## Usage Instructions
### Prerequisites
- Python 3.6 or higher
- NumPy
- Pandas
- Matplotlib
- TensorFlow 2.x
- Scikit-learn
- SymPy

### Installation
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install numpy pandas matplotlib tensorflow scikit-learn sympy
```

### Quick Start
1. Basic Neural Network Training:
```python
from neural_network_tensorflow import Sequential, Dense

# Create a simple neural network
model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=3, activation='softmax')
])

# Compile and train the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(X, y, epochs=100)
```

### More Detailed Examples
1. Model Evaluation and Selection:
```python
from model_evaluation_and_selection import train_test_split, StandardScaler

# Split dataset
x_train, x_cv, x_test = train_test_split(x, train_size=0.6)

# Scale features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_cv_scaled = scaler.transform(x_cv)
```

### Troubleshooting
Common Issues:
1. Memory Errors during Model Training
   - Reduce batch size
   - Decrease model complexity
   - Check for memory leaks in custom layers

2. Vanishing/Exploding Gradients
   - Use appropriate weight initialization
   - Add batch normalization layers
   - Adjust learning rate

Debug Mode:
```python
import tensorflow as tf
tf.debugging.set_log_device_placement(True)
```

## Data Flow
The framework implements a standard machine learning pipeline with data preprocessing, model training, and evaluation phases.

```ascii
Input Data → Preprocessing → Model Training → Evaluation → Prediction
    ↑          (Scaling)      (Training Set)   (CV Set)      ↓
    └──────────────────── Feedback Loop ─────────────────────┘
```

Key Component Interactions:
1. Data preprocessing transforms raw input into model-ready format
2. Model training optimizes parameters using specified algorithm
3. Evaluation metrics guide model selection and hyperparameter tuning
4. Cross-validation ensures model generalization
5. Prediction pipeline applies trained model to new data