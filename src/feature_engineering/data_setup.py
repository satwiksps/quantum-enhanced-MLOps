import torch                                    # The main PyTorch library for tensors.
from torchvision import datasets, transforms      # Provides access to datasets like MNIST and image tools.
from torch.utils.data import DataLoader, Subset # Helper classes for batching and subsetting data.

def get_data_loaders(batch_size=32, n_samples=500, img_size=14):
    """
    Prepares and returns data loaders for the MNIST dataset.
    The images are downscaled to speed up computation.
    """
    print(f"Preparing MNIST dataset. Downscaling images to {img_size}x{img_size}.")
    
    # --- Defines a pipeline of transformations for the input images ---
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),    # 1. Resize each image to the specified size.
        transforms.ToTensor(),                      # 2. Convert the image to a PyTorch tensor.
        transforms.Normalize((0.5,), (0.5,))        # 3. Normalize pixel values to be between -1 and 1.
    ])

    # --- Load the MNIST dataset from torchvision ---
    # Downloads the training dataset if not already present in the './data' folder.
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # Downloads the testing dataset.
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # --- Create smaller subsets of the data for faster execution ---
    train_indices = torch.arange(n_samples)       # Create a list of indices from 0 to n_samples-1.
    test_indices = torch.arange(n_samples)        # Create another list for the test set.

    # Create a subset using only the specified indices.
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    # --- Create DataLoader objects to manage batching and shuffling ---
    # The DataLoader provides an efficient way to iterate over data in batches.
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True) # shuffle=True is important for training.
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    print(f"Data loaders created with {len(train_subset)} training samples and {len(test_subset)} test samples.")
    return train_loader, test_loader              # Return the configured data loaders.