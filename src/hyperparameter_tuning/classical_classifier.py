import torch                                    # The main PyTorch library.
import torch.nn as nn                           # Provides neural network layers.
from torch.utils.data import DataLoader, TensorDataset # Helpers for batching data.
from sklearn.model_selection import train_test_split # A tool to split data into training and testing sets.
import numpy as np                              # For numerical operations.

class SimpleClassifier(nn.Module):
    """A simple feed-forward neural network for classification."""
    def __init__(self, input_dim, hidden_dim, output_dim=10, dropout_rate=0.5):
        super(SimpleClassifier, self).__init__()
        # --- Defines the sequence of layers in the neural network ---
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),       # 1. A fully connected layer.
            nn.ReLU(),                              # 2. An activation function.
            nn.Dropout(dropout_rate),               # 3. A regularization layer to prevent overfitting.
            nn.Linear(hidden_dim, output_dim)       # 4. The final output layer.
        )

    def forward(self, x):
        return self.network(x)                    # Defines the forward pass of data through the network.

def evaluate_model(hyperparams, quantum_features, labels):
    """Trains and evaluates the classifier for a given set of hyperparameters."""
    # --- Unpack the hyperparameters for this specific evaluation run ---
    hidden_dim = int(hyperparams['hidden_dim'])
    lr = hyperparams['lr']
    dropout = hyperparams['dropout']
    input_dim = quantum_features.shape[1]         # Get the feature dimension from the data itself.
    
    # --- Split the data into training (70%) and testing (30%) sets ---
    X_train, X_test, y_train, y_test = train_test_split(
        quantum_features, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # --- Prepare PyTorch datasets and data loaders for batching ---
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # --- Initialize the model and set up the training components ---
    model = SimpleClassifier(input_dim, hidden_dim, dropout_rate=dropout)
    criterion = nn.CrossEntropyLoss()             # The loss function for multi-class classification.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # The algorithm to update model weights.

    # --- Training Loop ---
    model.train()                                 # Set the model to training mode.
    for _ in range(5):                            # Loop over the dataset 5 times.
        for features, batch_labels in train_loader:
            optimizer.zero_grad()                 # Reset gradients.
            outputs = model(features)             # Get model predictions.
            loss = criterion(outputs, batch_labels) # Calculate the error.
            loss.backward()                       # Calculate gradients.
            optimizer.step()                      # Update weights.
            
    # --- Evaluation Loop ---
    model.eval()                                  # Set the model to evaluation mode (disables dropout).
    correct, total = 0, 0                         # Initialize counters.
    with torch.no_grad():                         # Disable gradient calculation for speed.
        for features, batch_labels in test_loader:
            outputs = model(features)             # Get model predictions.
            _, predicted = torch.max(outputs.data, 1) # Get the class with the highest score.
            total += batch_labels.size(0)         # Add the batch size to the total.
            correct += (predicted == batch_labels).sum().item() # Count correct predictions.
            
    return correct / total                        # Return the final accuracy score.