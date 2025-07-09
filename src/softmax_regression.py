import numpy as np
import sys
sys.path.append('.')
from src.data_loader import load_mnist

class SoftmaxRegression:
    """
    A multi-class Softmax Regression classifier implemented from scratch.

    This model is trained using batch gradient descent and supports L2 regularization.

    Attributes:
        learning_rate (float): The step size for gradient descent.
        n_iterations (int): The number of passes over the training dataset.
        regularization_lambda (float): The L2 regularization strength.
        weights (np.ndarray): The learned weights, shape (n_features, n_classes).
        bias (np.ndarray): The learned bias terms, shape (1, n_classes).
    """
    def __init__(self, learning_rate=0.1, n_iterations=1000, regularization_lambda=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization_lambda = regularization_lambda
        self.weights = None
        self.bias = None
        self.n_classes = None

    def _softmax(self, z):
        """
        Private helper to compute the softmax function.
        
        To prevent numerical instability (overflow), we subtract the max
        value of z from each element before exponentiating.
        """
        # z has shape (n_samples, n_classes)
        exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def _one_hot_encode(self, y, n_classes):
        """Private helper for one-hot encoding labels."""
        n_samples = len(y)
        y_one_hot = np.zeros((n_samples, n_classes))
        y_one_hot[np.arange(n_samples), y] = 1 #setting y_one_hot [row index, column (y -> anything bw 0 to 9)] to 1
        return y_one_hot

    def _compute_cost(self, y_one_hot, probabilities):
        """
        Computes the cross-entropy cost with L2 regularization.
        """
        n_samples = y_one_hot.shape[0]
        
        # Cross-entropy loss
        # Add a small epsilon (1e-9) for numerical stability to avoid log(0)
        log_loss = - (1 / n_samples) * np.sum(y_one_hot * np.log(probabilities + 1e-9))
        
        # L2 Regularization term
        l2_reg_cost = (self.regularization_lambda / (2 * n_samples)) * np.sum(np.square(self.weights))
        
        total_cost = log_loss + l2_reg_cost
        return total_cost

    def fit(self, X, y):
        """
        Trains the softmax regression model using the training data.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray): Multi-class target values, shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.n_classes = len(np.unique(y))
        
        # Initialize weights and bias
        self.weights = np.zeros((n_features, self.n_classes))
        self.bias = np.zeros((1, self.n_classes))
        
        # One-hot encode the labels
        y_one_hot = self._one_hot_encode(y, self.n_classes)
        
        # Gradient Descent
        for i in range(self.n_iterations):
            # --- Forward Pass ---
            # 1. Calculate linear scores (logits)
            logits = np.dot(X, self.weights) + self.bias
            # 2. Get probabilities via softmax
            probabilities = self._softmax(logits)
            
            # --- Gradient Calculation ---
            # The gradient of the cross-entropy loss w.r.t. logits is surprisingly simple
            error = probabilities - y_one_hot
            
            # 3. Gradients for weights and bias
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error, axis=0, keepdims=True)
            
            # 4. Add L2 regularization to the weight gradient
            dw += (self.regularization_lambda / n_samples) * self.weights
            
            # --- Update Rule ---
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Optional: Print cost every 100 iterations to monitor training
            if i % 100 == 0:
                cost = self._compute_cost(y_one_hot, probabilities)
                print(f"Iteration {i}: Cost = {cost:.4f}")

    def predict(self, X):
        """
        Makes a multi-class prediction by choosing the class with the highest probability.

        Args:
            X (np.ndarray): Data to predict on, shape (n_samples, n_features).

        Returns:
            np.ndarray: The predicted class labels, shape (n_samples,).
        """
        if self.weights is None or self.bias is None:
            raise RuntimeError("The model has not been trained yet. Call fit() first.")
        
        logits = np.dot(X, self.weights) + self.bias
        probabilities = self._softmax(logits)
        # Return the index of the maximum probability for each sample
        return np.argmax(probabilities, axis=1)


# Test Block
if __name__ == "__main__":
    print("Loading MNIST data...")
    X_train, X_test, y_train, y_test = load_mnist()

    print("\nInitializing Softmax Regression Classifier...")
    # For better accuracy, you might increase n_iterations to 1000-2000
    # and tune the learning_rate and regularization_lambda.
    model = SoftmaxRegression(learning_rate=0.1, n_iterations=500, regularization_lambda=0.01)

    print("\nStarting training...")
    model.fit(X_train, y_train)
    print("Training complete.")

    print("\nEvaluating model on the test set...")
    predictions = model.predict(X_test)
    
    accuracy = np.mean(predictions == y_test)
    print(f"\nClassification Accuracy on Test Set: {accuracy * 100:.2f}%")