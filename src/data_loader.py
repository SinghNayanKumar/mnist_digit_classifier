import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os

def load_mnist():

    """
    Loads the MNIST dataset from the OpenML repository. This data contains both the image data and the corresponding labels.

    This function handles loading the data, normalizing pixel values to the [0, 1]
    range, and flattening the 28x28 images into 784-dimensional vectors.

    Args:
        path (str): The file path to the mnist.npz file.

    Returns:
        tuple: A tuple containing four numpy arrays:
               X_train, X_test, y_train, y_test
               - X_train: (56000, 784) numpy array of training image data.
               - y_train: (56000,) numpy array of training labels.
               - X_test: (14000, 784) numpy array of test image data.
               - y_test: (14000,) numpy array of test labels.
    
    """

    
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.to_numpy().astype(np.float32) / 255.0
    y = mnist.target.to_numpy().astype(np.int64)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Test Block
if __name__ == "__main__":
    print("Testing the data loader...")
    try:
        X_train, X_test, y_train, y_test = load_mnist()

        print("\nData loaded successfully!")
        print("--- Shapes ---")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")

        print("\n--- Data Types ---")
        print(f"X_train dtype: {X_train.dtype}")
        print(f"y_train dtype: {y_train.dtype}")

        print("\n--- Data Range (Normalization Check) ---")
        print(f"X_train min value: {np.min(X_train):.1f}")
        print(f"X_train max value: {np.max(X_train):.1f}")

        print("\n--- Label Information ---")
        print(f"Unique labels: {np.unique(y_train)}")
        print(f"Number of unique labels: {len(np.unique(y_train))}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")