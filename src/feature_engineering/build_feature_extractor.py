import torch                                    # The main PyTorch library for tensors and neural networks.
import torch.nn as nn                           # Provides neural network layers (Linear, Conv2d, etc.).
import torch.optim as optim                     # Provides optimization algorithms like Adam.
import numpy as np                              # For numerical operations, especially saving weights.
import os                                       # For handling file paths and creating directories.
from tqdm import tqdm                           # For creating smart progress bars during training.
import mlflow                                   # For logging metrics and artifacts.

from .data_setup import get_data_loaders         # Imports the function to prepare the MNIST data.
from .classical_components import Encoder, Decoder # Imports the classical parts of the autoencoder.
from .quantum_circuits import get_quantum_torch_layer # Imports the function to create the quantum layer.

class HybridAutoencoder(nn.Module):
    """The full Hybrid Quantum-Classical Autoencoder."""
    def __init__(self, encoder, quantum_layer, decoder):
        super(HybridAutoencoder, self).__init__()
        self.encoder = encoder                    # The classical CNN part that compresses the image.
        self.quantum_layer = quantum_layer        # The quantum circuit that processes the compressed data.
        self.decoder = decoder                    # The classical part that reconstructs the image.

    def forward(self, x):
        latent_vec = self.encoder(x)              # Compress image into a latent vector.
        quantum_output = self.quantum_layer(latent_vec) # Process latent vector through the PQC.
        reconstructed_image = self.decoder(quantum_output) # Reconstruct image from PQC output.
        return reconstructed_image

# The function now accepts the master config object
def run_feature_engineering(config):
    print("--- MLOps Stage 1: Building Quantum-Native Feature Extractor ---")
    
    # --- Read parameters from the config file ---
    cfg = config['stage_1_feature_engineering']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Check for GPU, otherwise use CPU.
    print(f"Using device: {device}")
    
    # --- Prepare the dataset using settings from the config ---
    train_loader, _ = get_data_loaders(
        batch_size=cfg['stage_1_batch_size'], 
        n_samples=cfg['stage_1_n_samples'], 
        img_size=cfg['stage_1_img_size']
    )
    
    # --- Build the Hybrid Model Components ---
    encoder = Encoder(cfg['stage_1_latent_dim'], cfg['stage_1_img_size']).to(device) # Create the classical encoder.
    quantum_layer = get_quantum_torch_layer(cfg['stage_1_latent_dim']).to(device) # Create the quantum layer.
    decoder = Decoder(cfg['stage_1_latent_dim'], cfg['stage_1_img_size']).to(device) # Create the classical decoder.
    model = HybridAutoencoder(encoder, quantum_layer, decoder).to(device) # Assemble the full hybrid model.
    
    # --- Set up the Training Process ---
    criterion = nn.MSELoss()                        # The loss function measures reconstruction error.
    optimizer = optim.Adam(model.parameters(), lr=cfg['stage_1_learning_rate']) # The algorithm to update model weights.
    
    print("\nStarting Hybrid Autoencoder training...")
    # --- Main Training Loop ---
    for epoch in range(cfg['stage_1_epochs']):      # Loop over the dataset multiple times.
        model.train()                               # Set the model to training mode.
        running_loss = 0.0                          # Initialize loss counter for the epoch.
        
        # Use tqdm for a visual progress bar over the data batches.
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['stage_1_epochs']}")
        
        for images, _ in progress_bar:              # Loop through each batch of images.
            images = images.to(device)              # Move data to the selected device (CPU/GPU).
            optimizer.zero_grad()                   # Reset gradients from the previous step.
            outputs = model(images)                 # Pass images through the model to get reconstructions.
            loss = criterion(outputs, images)       # Calculate the error between output and input.
            loss.backward()                         # Calculate gradients (backpropagation).
            optimizer.step()                        # Update the model's weights.
            running_loss += loss.item()             # Add the batch loss to the total.
            progress_bar.set_postfix(loss=loss.item()) # Update the progress bar display.
            
        avg_loss = running_loss / len(train_loader) # Calculate the average loss for the epoch.
        print(f"Epoch [{epoch+1}/{cfg['stage_1_epochs']}], Average Training Loss: {avg_loss:.4f}")
        mlflow.log_metric(f"stage_1_epoch_{epoch+1}_loss", avg_loss) # Log the epoch loss to MLflow.

    print("\nTraining finished.")
    
    # --- Save the Trained Model Artifacts ---
    save_dir = "saved_models/feature_extractor"     # Define the directory to save models.
    os.makedirs(save_dir, exist_ok=True)            # Create the directory if it doesn't exist.
    
    encoder_path = os.path.join(save_dir, "hae_encoder.pth") # Define the file path for the encoder.
    torch.save(model.encoder.state_dict(), encoder_path) # Save the encoder's learned weights.
    print(f"Trained classical encoder saved to {encoder_path}")
    
    pqc_weights_path = os.path.join(save_dir, "hae_pqc_weights.npy") # Define the path for the quantum weights.
    pqc_weights = model.quantum_layer.weight.cpu().detach().numpy() # Get the PQC weights from the model.
    np.save(pqc_weights_path, pqc_weights)          # Save the PQC weights as a numpy array.
    print(f"Trained PQC weights saved to {pqc_weights_path}")
    
    # --- Log Artifacts to MLflow ---
    print("Logging model artifacts to MLflow...")
    # Save a copy of the encoder to the MLflow run.
    mlflow.log_artifact(encoder_path, artifact_path="stage_1_feature_extractor")
    # Save a copy of the PQC weights to the MLflow run.
    mlflow.log_artifact(pqc_weights_path, artifact_path="stage_1_feature_extractor")
    
    print("\n--- Feature Engineering Stage Complete ---")