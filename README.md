# Machine Learning & Deep Learning

## Overview

This project is a modular framework for building, training, and evaluating machine learning models, with a strong focus on neural networks. It provides both custom (NumPy-based) and TensorFlow-based implementations, supporting regression and classification tasks, model selection, hyperparameter tuning, and performance evaluation

---

## Features

- **Regression (Custom):**  
  - Linear and logistic regression from scratch
  - Custom cost functions and gradient descent

- **Neural Networks from Scratch (NumPy):**  
  - Build neural networks without deep learning libraries
  - Single and multi-layer architectures

- **Neural Networks with TensorFlow:**  
  - Deep neural networks using TensorFlow/Keras
  - Support for both regression and classification tasks

- **Symbolic Differentiation:**  
  - Symbolic derivatives using SymPy for understanding gradients and backpropagation

- **Optimization Algorithms:**  
  - Gradient Descent (custom)
  - Adam Optimizer (TensorFlow)

- **Model Evaluation & Selection:**  
  - Automated splitting into training, cross-validation, and test sets
  - Model selection based on cross-validation performance
  - Visualization of model performance

- **Data Preprocessing:**  
  - Feature scaling (StandardScaler)
  - Polynomial feature generation

---

## Repository Structure

```
.
├── adam_optimization_algorithm.py     # Adam optimizer demo with Keras
├── derivatives.py                     # Symbolic differentiation with SymPy
├── model_evaluation_and_selection_(neural_network).py    # NN model evaluation & selection
├── model_evaluation_and_selection.py  # General model evaluation & selection (regression)
├── neural_network_1.py                # Basic NumPy NN (1 hidden layer)
├── neural_network_2.py                # NumPy NN (2 hidden layers, modular)
├── neural_network_3.py                # Enhanced NumPy NN (vectorized)
├── neural_network_tensoflow.py        # TensorFlow/Keras NN implementation
├── regression.py                      # Linear & logistic regression (custom, with GD)
├── data/
│   ├── model_evaluation_and_selection_dataset.csv
│   └── model_evaluation_and_selection_dataset(classification).csv
└── .venv/, .git/, .vscode/            # Environment, version control, editor settings
```

---

## Installation

**Prerequisites:**
- Python 3.6+
- NumPy, Pandas, Matplotlib, TensorFlow 2.x, Scikit-learn, SymPy

**Setup:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

pip install numpy pandas matplotlib tensorflow scikit-learn sympy
```

---

## Usage

Follow this recommended order to understand and experiment with the project step by step:

### 1. Regression (Custom Implementation)
- Explore linear and logistic regression, cost functions, and gradient descent from scratch.
- File: `regression.py`

### 2. Neural Networks from Scratch (NumPy)
- Build neural networks without any deep learning libraries.
- Files (in order):
  - `neural_network_1.py` (single hidden layer, basic)
  - `neural_network_2.py` (two hidden layers, modular)
  - `neural_network_3.py` (vectorized, enhanced)

### 3. Neural Networks with TensorFlow
- Implement and train neural networks using TensorFlow/Keras.
- File: `neural_network_tensoflow.py`

### 4. Symbolic Differentiation
- Use SymPy for symbolic derivatives, useful for understanding gradients and backpropagation.
- File: `derivatives.py`

### 5. Adam Optimization Algorithm (TensorFlow)
- Demonstrate the Adam optimizer in a neural network context.
- File: `adam_optimization_algorithm.py`

### 6. Model Evaluation & Selection (Regression)
- Learn about model selection, polynomial features, and evaluation for regression tasks.
- File: `model_evaluation_and_selection.py`

### 7. Model Evaluation & Selection (Neural Networks)
- Evaluate and select neural network models for both regression and classification tasks.
- File: `model_evaluation_and_selection_(neural_network).py`

---

See each file for code, comments, and examples. The data used for evaluation is in the `data/` directory.

---

## Data

- `data/model_evaluation_and_selection_dataset.csv`: Regression dataset
- `data/model_evaluation_and_selection_dataset(classification).csv`: Classification dataset

---

## Visualization

- The framework includes plotting for data, model predictions, and error curves (using Matplotlib).

---

## Troubleshooting

- **Memory Errors:** Reduce batch size, decrease model complexity, or check for memory leaks.
- **Vanishing/Exploding Gradients:** Use proper weight initialization, batch normalization, or adjust learning rate.
- **Debugging TensorFlow:**  
  ```python
  import tensorflow as tf
  tf.debugging.set_log_device_placement(True)
  ```

---

## Extending the Framework

- Add new models by following the modular structure.
- Integrate new datasets by placing them in the `data/` directory and updating the relevant scripts.

---

## License

This project is for educational and research purposes.

---
