import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import os

# Add project root to the path to allow direct script execution
sys.path.append('.')
from src.data_loader import load_mnist

# 1. Define the Neural Network Architecture
class SimpleNN(nn.Module):
    """
    A simple feed-forward neural network model using PyTorch.
    
    Architecture:
    - Input layer (784 features)
    - Hidden layer 1 (128 neurons) with ReLU activation
    - Hidden layer 2 (64 neurons) with ReLU activation
    - Output layer (10 neurons for 10 classes)
    """
    def __init__(self, input_size=784, hidden_size1=128, hidden_size2=64, num_classes=10):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, num_classes)
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.
        The nn.CrossEntropyLoss in the training loop will apply softmax implicitly.
        """
        return self.layers(x)

# 2. Create a Wrapper Class to Handle Training and Prediction
class NeuralNetwork:
    """
    A wrapper for the PyTorch SimpleNN model to handle training, prediction,
    and model persistence in a user-friendly way.
    """
    def __init__(self, learning_rate=0.001, n_epochs=10, batch_size=64):
        # Hyperparameters
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Device configuration (use GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize model, loss function, and optimizer
        self.model = SimpleNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X_train, y_train):
        """
        Trains the neural network model.

        Args:
            X_train (np.ndarray): Training feature data.
            y_train (np.ndarray): Training labels.
        """
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).long()

        # Create DataLoader for batching and shuffling
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Set the model to training mode
        self.model.train()

        history = {'loss': []}

        print("\n--- Starting Training ---")
        for epoch in range(self.n_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                # Move tensors to the configured device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # 1. Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # 2. Forward pass
                outputs = self.model(inputs)
                
                # 3. Calculate loss
                loss = self.criterion(outputs, labels)
                
                # 4. Backward pass (backpropagation)
                loss.backward()
                
                # 5. Update weights
                self.optimizer.step()
                
                running_loss += loss.item()

            avg_epoch_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{self.n_epochs}], Loss: {avg_epoch_loss:.4f}")
        print("--- Training Complete ---")
        return history

    def predict(self, X_test):
        """
        Makes predictions on new data.

        Args:
            X_test (np.ndarray): Test feature data.

        Returns:
            np.ndarray: Predicted labels.
        """
        # Convert numpy array to PyTorch tensor
        X_test_tensor = torch.from_numpy(X_test).float().to(self.device)
        
        # Set the model to evaluation mode (important for layers like Dropout, BatchNorm)
        self.model.eval()
        
        # Disable gradient calculations for inference
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            # Get the index of the max log-probability (the predicted class)
            _, predicted_classes = torch.max(outputs.data, 1)
        
        # Move predictions to CPU and convert to NumPy array before returning
        return predicted_classes.cpu().numpy()

    def save_model(self, path="models/neural_network.pth"):
        """Saves the trained model's state dictionary."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="models/neural_network.pth"):
        """Loads a model's state dictionary from a file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {path}")


# Test Block
if __name__ == "__main__":
    print("Loading MNIST data...")
    X_train, X_test, y_train, y_test = load_mnist()
    
    print("\nInitializing Neural Network...")
    nn_classifier = NeuralNetwork(learning_rate=0.001, n_epochs=10, batch_size=64)
    
    # Train the model
    nn_classifier.fit(X_train, y_train)
    
    # Evaluate the model
    print("\nEvaluating model on the test set...")
    predictions = nn_classifier.predict(X_test)
    
    accuracy = np.mean(predictions == y_test)
    print(f"\nClassification Accuracy on Test Set: {accuracy * 100:.2f}%")
    
    # Example of saving and loading the model
    model_path = "models/nn_model.pth"
    nn_classifier.save_model(model_path)
    
    # Create a new instance and load the saved model to verify
    print("\nTesting model loading...")
    new_nn = NeuralNetwork()
    new_nn.load_model(model_path)
    new_predictions = new_nn.predict(X_test)
    new_accuracy = np.mean(new_predictions == y_test)
    print(f"Accuracy of loaded model: {new_accuracy * 100:.2f}%")