import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

class SimpleClassifier(nn.Module):
    """A simple feed-forward neural network for classification."""
    def __init__(self, input_dim, hidden_dim, output_dim=10, dropout_rate=0.5):
        super(SimpleClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

def evaluate_model(hyperparams, quantum_features, labels):
    """Trains and evaluates the classifier for a given set of hyperparameters."""
    hidden_dim = int(hyperparams['hidden_dim'])
    lr = hyperparams['lr']
    dropout = hyperparams['dropout']
    input_dim = quantum_features.shape[1]
    
    X_train, X_test, y_train, y_test = train_test_split(
        quantum_features, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = SimpleClassifier(input_dim, hidden_dim, dropout_rate=dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(5):
        for features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for features, batch_labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
    return correct / total