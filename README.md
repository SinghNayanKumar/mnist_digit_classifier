# MNIST Digit Classifier from Scratch

This project implements several machine learning models from scratch using NumPy to classify handwritten digits from the famous MNIST dataset. The goal is to demonstrate the fundamental concepts behind these algorithms without relying on high-level machine learning libraries like Scikit-learn or TensorFlow for the core model logic.

eg. [MNIST Digits] (https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Modules Explained](#modules-explained)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Report](#report)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Data Loading:** A custom data loader for the `mnist.npz` dataset.
- **Model Implementations (from scratch):**
    - **Logistic Regression:** A binary-style (One-vs-Rest) logistic regression classifier.
    - **Softmax Regression:** A multi-class logistic regression classifier.
    - **Neural Network:** A simple feed-forward neural network with configurable layers and activation functions.
- **Utilities:** Helper functions for metrics, one-hot encoding, and more.
- **Analysis:** An exploratory data analysis notebook and a detailed PDF report on model performance.

## Project Structure
```
mnist_digit_classifier/
├── data/
│   └── mnist.npz
├── src/
│   ├── data_loader.py
│   ├── logistic_regression.py
│   ├── softmax_regression.py
│   ├── neural_network.py
│   ├── utils.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── main.py
├── README.md
└── report.pdf
```

## Setup and Installation

Follow these steps to set up the project environment.

**1. Clone the repository:**
```bash
git clone <your-repository-url>
cd mnist_digit_classifier
```

**2. Create and activate a virtual environment (recommended):**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

**3. Install dependencies:**
This project relies on NumPy for numerical operations and Matplotlib for plotting. You will also need Jupyter to run the notebook.
```bash
pip install numpy matplotlib jupyterlab
```
*(You can also create a `requirements.txt` file with these packages for easier installation.)*

**4. Download the dataset:**
The `mnist.npz` file is a NumPy archive of the dataset. If you don't have it, you can download it from a reliable source (like Keras's dataset storage) and place it in the `data/` directory.

Here is a Python snippet you can run from the project root to download the data:
```python
import os
import requests
import numpy as np

DATA_DIR = "data"
DATA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
DATA_PATH = os.path.join(DATA_DIR, "mnist.npz")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(DATA_PATH):
    print("Downloading MNIST dataset...")
    response = requests.get(DATA_URL)
    with open(DATA_PATH, 'wb') as f:
        f.write(response.content)
    print("Download complete.")
else:
    print("MNIST dataset already exists.")
```

## Usage

The main entry point for training and evaluating the models is `main.py`. You can select which model to run using the `--model` command-line argument.

### Available Models
*   `logistic_regression`
*   `softmax_regression`
*   `neural_network`

### Example Commands

**Train and evaluate the Softmax Regression model:**
```bash
python main.py --model softmax_regression
```

**Train and evaluate the Neural Network model:**
```bash
python main.py --model neural_network
```

**Train and evaluate the Logistic Regression model:**
```bash
python main.py --model logistic_regression
```

The script will load the data, train the chosen model, and print the final accuracy on the test set.

## Modules Explained

Here is a brief overview of the key Python files in the `src/` directory.

- **`src/data_loader.py`**: Contains the function `load_mnist_data()` which loads the training and testing data from `data/mnist.npz`. It handles data normalization and reshaping.

- **`src/logistic_regression.py`**: Implements a One-vs-Rest (OvR) logistic regression classifier. It trains 10 separate binary classifiers, one for each digit.

- **`src/softmax_regression.py`**: Implements Softmax Regression (also known as Multinomial Logistic Regression). This is a single model that directly handles multi-class classification.

- **`src/neural_network.py`**: Implements a simple feed-forward neural network. The architecture (number of layers, neurons) can be configured within the file. It uses backpropagation to train.

- **`src/utils.py`**: A collection of utility functions used across the project, such as:
    - `one_hot_encode()`: Converts integer labels to one-hot vectors.
    - `accuracy()`: Calculates the classification accuracy.
    - Other potential helpers for plotting or data manipulation.

## Exploratory Data Analysis

The `notebooks/exploratory_analysis.ipynb` Jupyter Notebook contains an initial analysis of the MNIST dataset. It visualizes sample digits, checks the class distribution, and explores pixel intensity distributions.

To run the notebook:
```bash
jupyter lab notebooks/exploratory_analysis.ipynb
```

## Report

For a detailed discussion of the model architectures, training processes, results, and a comparison of their performance, please see the `report.pdf` file included in this repository.

## Contributing

Contributions are welcome! If you would like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.