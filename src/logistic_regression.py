import numpy as np
import sys
sys.path.append('.') 
from src.data_loader import load_mnist
import copy

class LogisticRegression:
    """
    A binary Logistic Regression classifier implemented from scratch.

    This model is trained using batch gradient descent and supports L2 regularization.

    Attributes:
        learning_rate (float): The step size for gradient descent.
        n_iterations (int): The number of passes over the training dataset.
        regularization_lambda (float): The L2 regularization strength.
        weights (np.ndarray): The learned weights for the features.
        bias (float): The learned bias term.
    """
    def __init__(self, learning_rate=0.1, n_iterations=1000, regularization_lambda=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization_lambda = regularization_lambda
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """Private helper to compute the sigmoid function."""
        # Clip z to avoid overflow in np.exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):  
        """
        Trains the logistic regression model using the training data.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray): Binary target values, shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Add L2 regularization term to weight gradient (bias is not regularized)
            dw += (self.regularization_lambda / n_samples) * self.weights

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print cost for monitoring
            if i % 100 == 0:
                cost = self._compute_cost(X, y, y_predicted)
                print(f"Iteration {i}: Cost = {cost:.4f}")

    def _compute_cost(self, X, y, y_predicted):
        """Computes the binary cross-entropy cost with L2 regularization."""
        n_samples = len(y)
        # Binary cross-entropy
        log_loss = -1/n_samples * np.sum(y * np.log(y_predicted + 1e-9) + (1-y) * np.log(1 - y_predicted + 1e-9))
        # L2 Regularization term
        l2_reg = (self.regularization_lambda / (2 * n_samples)) * np.sum(np.square(self.weights))
        return log_loss + l2_reg


    def predict_proba(self, X):
        """
        Returns the probability that each sample belongs to the positive class.

        Args:
            X (np.ndarray): Data to predict on, shape (n_samples, n_features).

        Returns:
            np.ndarray: Probabilities for the positive class, shape (n_samples,).
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)


class OneVsRestClassifier:
    """
    A One-vs-Rest (OvR) classifier that uses a binary classifier (e.g., LogisticRegression)
    to perform multi-class classification.
    """
    def __init__(self, base_classifier, n_classes=10):
        self.base_classifier = base_classifier
        self.n_classes = n_classes
        self.classifiers = []

    def fit(self, X, y):
        """
        Trains one binary classifier for each class.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray): Multi-class target values, shape (n_samples,).
        """
        self.classifiers = [] # Clear any previous classifiers

        # Train one classifier for each class
        for i in range(self.n_classes):
            print(f"--- Training Classifier for Digit {i} ---")
            # Create a copy of the base classifier
            classifier = copy.deepcopy(self.base_classifier)
            
            # Create a binary target vector: 1 for the current class, 0 for all others
            y_binary = (y == i).astype(int)
            
            classifier.fit(X, y_binary)
            self.classifiers.append(classifier)
            print(f"--- Classifier for Digit {i} Trained ---\n")

    def predict(self, X):
        """
        Makes a multi-class prediction by choosing the class with the highest probability.

        Args:
            X (np.ndarray): Data to predict on, shape (n_samples, n_features).

        Returns:
            np.ndarray: The predicted class labels, shape (n_samples,).
        """
        # Get probabilities from each classifier
        probabilities = np.zeros((X.shape[0], self.n_classes))
        for i, classifier in enumerate(self.classifiers):
            probabilities[:, i] = classifier.predict_proba(X)
        
        # Return the class with the highest probability for each sample
        return np.argmax(probabilities, axis=1)


# Test Block
if __name__ == "__main__":
    print("Loading MNIST data...")
    X_train, X_test, y_train, y_test = load_mnist()
    
    print("\nInitializing One-vs-Rest Logistic Regression Classifier...")
    # Use fewer iterations for a quick test run. For better accuracy, use ~1000.
    lr_model = LogisticRegression(learning_rate=0.1, n_iterations=200, regularization_lambda=0.01)
    ovr_classifier = OneVsRestClassifier(base_classifier=lr_model, n_classes=10)

    print("\nStarting training...")
    ovr_classifier.fit(X_train, y_train)
    print("Training complete.")

    print("\nEvaluating model on the test set...")
    predictions = ovr_classifier.predict(X_test)
    
    accuracy = np.mean(predictions == y_test)
    print(f"\nClassification Accuracy on Test Set: {accuracy * 100:.2f}%")