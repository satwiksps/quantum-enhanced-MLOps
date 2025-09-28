import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_data_loaders(batch_size=32, n_samples=500, img_size=14):
    """
    Prepares and returns data loaders for the MNIST dataset.
    The images are downscaled to speed up computation.
    """
    print(f"Preparing MNIST dataset. Downscaling images to {img_size}x{img_size}.")
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_indices = torch.arange(n_samples)
    test_indices = torch.arange(n_samples)

    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    print(f"Data loaders created with {len(train_subset)} training samples and {len(test_subset)} test samples.")
    return train_loader, test_loader