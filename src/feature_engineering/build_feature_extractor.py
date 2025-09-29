import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
import mlflow

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

def run_feature_engineering(config):
    print("--- MLOps Stage 1: Building Quantum-Native Feature Extractor ---")
    cfg = config['stage_1_feature_engineering']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, _ = get_data_loaders(
        batch_size=cfg['stage_1_batch_size'], 
        n_samples=cfg['stage_1_n_samples'], 
        img_size=cfg['stage_1_img_size']
    )
    encoder = Encoder(cfg['stage_1_latent_dim'], cfg['stage_1_img_size']).to(device)
    quantum_layer = get_quantum_torch_layer(cfg['stage_1_latent_dim']).to(device)
    decoder = Decoder(cfg['stage_1_latent_dim'], cfg['stage_1_img_size']).to(device)
    model = HybridAutoencoder(encoder, quantum_layer, decoder).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['stage_1_learning_rate'])
    
    print("\nStarting Hybrid Autoencoder training...")
    for epoch in range(cfg['stage_1_epochs']):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['stage_1_epochs']}")
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
        print(f"Epoch [{epoch+1}/{cfg['stage_1_epochs']}], Average Training Loss: {avg_loss:.4f}")
        mlflow.log_metric(f"stage_1_epoch_{epoch+1}_loss", avg_loss)

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
    
    print("Logging model artifacts to MLflow...")
    mlflow.log_artifact(encoder_path, artifact_path="stage_1_feature_extractor")
    mlflow.log_artifact(pqc_weights_path, artifact_path="stage_1_feature_extractor")
    
    print("\n--- Feature Engineering Stage Complete ---")