import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import torch

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Generates and plots a confusion matrix.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted labels.
        class_names (list): List of class names for axis labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def print_classification_report(y_true, y_pred, class_names):
    """
    Prints the classification report from scikit-learn.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted labels.
        class_names (list): List of class names for the report.
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:\n")
    print(report)

def plot_learning_curves(history, title="Learning Curves"):
    """
    Plots the learning curves for training loss.
    Assumes history is a dictionary like {'loss': [val1, val2, ...]}

    Args:
        history (dict): A dictionary containing loss values per epoch.
        title (str): The title for the plot.
    """
    if not history or 'loss' not in history:
        print("History object is empty or does not contain 'loss' key. Skipping plot.")
        return
        
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(X, y_true, y_pred, class_names, num_samples=25, show_incorrect=False):
    """
    Visualizes model predictions on a grid of images.

    Args:
        X (np.ndarray): Image data (flattened, shape (n_samples, 784)).
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        class_names (list): List of class names.
        num_samples (int): Number of samples to display.
        show_incorrect (bool): If True, shows only incorrect predictions.
    """
    plt.figure(figsize=(12, 12))
    
    if show_incorrect:
        # Find indices of incorrect predictions
        incorrect_indices = np.where(y_pred != y_true)[0]
        if len(incorrect_indices) == 0:
            print("No incorrect predictions to show!")
            return
        # Select random samples from the incorrect ones
        sample_indices = np.random.choice(incorrect_indices, size=min(num_samples, len(incorrect_indices)), replace=False)
        plot_title = "Incorrect Predictions"
    else:
        # Select random samples from all predictions
        sample_indices = np.random.choice(len(X), size=num_samples, replace=False)
        plot_title = "Model Predictions"

    for i, idx in enumerate(sample_indices):
        plt.subplot(5, 5, i + 1)
        # Reshape the flattened image to 28x28
        image = X[idx].reshape(28, 28)
        plt.imshow(image, cmap='gray')
        
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        
        color = 'green' if true_label == pred_label else 'red'
        
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis('off')
        
    plt.suptitle(plot_title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def evaluate_model(model, X_test, y_test, class_names):
    """
    A comprehensive evaluation function that runs all standard evaluations.

    Args:
        model: A trained model instance with a .predict() method.
        X_test (np.ndarray): Test feature data.
        y_test (np.ndarray): Test labels.
        class_names (list): List of class names.
    """
    print(f"--- Evaluating Model: {model.__class__.__name__} ---")
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%\n")
    
    # Print detailed classification report
    print_classification_report(y_test, y_pred, class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, class_names)
    
    # Visualize some correct predictions
    plot_predictions(X_test, y_test, y_pred, class_names, show_incorrect=False)

    # Visualize some incorrect predictions
    plot_predictions(X_test, y_test, y_pred, class_names, show_incorrect=True)