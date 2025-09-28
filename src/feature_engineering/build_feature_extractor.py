import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm

from .data_setup import get_data_loaders
from .classical_components import Encoder, Decoder
from .quantum_circuits import get_quantum_torch_layer

class HybridAutoencoder(nn.Module):
    """The full Hybrid Quantum-Classical Autoencoder."""
    def __init__(self, encoder, quantum_layer, decoder):
        super(HybridAutoencoder, self).__init__()
        self.encoder = encoder
        self.quantum_layer = quantum_layer
        self.decoder = decoder

    def forward(self, x):
        latent_vec = self.encoder(x)
        quantum_output = self.quantum_layer(latent_vec)
        reconstructed_image = self.decoder(quantum_output)
        return reconstructed_image

def run_feature_engineering(latent_dim=4, epochs=5, lr=0.001, batch_size=32, n_samples=600, img_size=14):
    """Main function to orchestrate the training of the Hybrid Autoencoder."""
    print("--- MLOps Stage 1: Building Quantum-Native Feature Extractor ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, _ = get_data_loaders(batch_size, n_samples, img_size)
    
    encoder = Encoder(latent_dim, img_size).to(device)
    quantum_layer = get_quantum_torch_layer(latent_dim).to(device)
    decoder = Decoder(latent_dim, img_size).to(device)
    model = HybridAutoencoder(encoder, quantum_layer, decoder).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("\nStarting Hybrid Autoencoder training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, _ in progress_bar:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Training Loss: {avg_loss:.4f}")

    print("\nTraining finished.")

    save_dir = "saved_models/feature_extractor"
    os.makedirs(save_dir, exist_ok=True)
    
    encoder_path = os.path.join(save_dir, "hae_encoder.pth")
    torch.save(model.encoder.state_dict(), encoder_path)
    print(f"Trained classical encoder saved to {encoder_path}")
    
    pqc_weights_path = os.path.join(save_dir, "hae_pqc_weights.npy")
    pqc_weights = model.quantum_layer.weight.cpu().detach().numpy()
    np.save(pqc_weights_path, pqc_weights)
    print(f"Trained PQC weights saved to {pqc_weights_path}")
    
    print("\n--- Feature Engineering Stage Complete ---")