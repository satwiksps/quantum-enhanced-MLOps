import matplotlib                               # The main plotting library.
matplotlib.use('Agg')                           # Sets the backend to prevent pop-up windows, important for servers.
import matplotlib.pyplot as plt                 # The primary interface for creating plots.
import torch                                    # The main PyTorch library.
import numpy as np                              # For numerical operations.
from sklearn.decomposition import PCA           # The algorithm for dimensionality reduction.

# --- Local Project Imports ---
from src.feature_engineering.data_setup import get_data_loaders
from src.feature_engineering.classical_components import Encoder
from src.feature_engineering.quantum_circuits import get_quantum_torch_layer
from src.hyperparameter_tuning.tune_with_qaoa import generate_quantum_features

# The function now accepts the master config object to get its parameters.
def create_feature_space_plot(config):
    print("\n[VISUALIZATION] Generating plot for Stage 1: Feature Space...")
    
    # --- Read parameters from the config file ---
    cfg1 = config['stage_1_feature_engineering']
    device = torch.device("cpu")                # Set the device to CPU for this script.
    
    # --- Load the saved model artifacts created by the main pipeline ---
    encoder = Encoder(cfg1['stage_1_latent_dim'], cfg1['stage_1_img_size']) # Initialize the encoder architecture.
    encoder.load_state_dict(torch.load("saved_models/feature_extractor/hae_encoder.pth")) # Load its saved weights.
    encoder.to(device)                          # Move the model to the CPU.
    
    quantum_layer = get_quantum_torch_layer(cfg1['stage_1_latent_dim']) # Initialize the quantum layer.
    pqc_weights = np.load("saved_models/feature_extractor/hae_pqc_weights.npy") # Load its saved weights.
    quantum_layer.weight = torch.nn.Parameter(torch.Tensor(pqc_weights)) # Assign the weights to the layer.
    quantum_layer.to(device)                    # Move the layer to the CPU.

    # --- Generate the data needed for this specific visualization ---
    vis_loader, _ = get_data_loaders(
        batch_size=cfg1['stage_1_n_samples'], 
        n_samples=cfg1['stage_1_n_samples'], 
        img_size=cfg1['stage_1_img_size']
    )
    # Use the loaded, trained models to generate the quantum features.
    features, labels = generate_quantum_features(encoder, quantum_layer, vis_loader, device)
    
    # --- Create the Plot ---
    # The 4D features are hard to see, so PCA projects them down to 2D.
    pca = PCA(n_components=2)                   # Initialize the PCA algorithm to find the 2 most important dimensions.
    features_2d = pca.fit_transform(features)   # Transform the 4D data into 2D data.
    
    fig, ax = plt.subplots(figsize=(10, 8))     # Create a new figure and axis for plotting.
    # Create a scatter plot, coloring each point based on its true digit label (0-9).
    scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    
    # --- Add Final Touches and Save the Image ---
    legend1 = ax.legend(*scatter.legend_elements(), title="Digits") # Create a legend based on the colors.
    ax.add_artist(legend1)                      # Add the legend to the plot.
    ax.set_title("Quantum-Native Feature Space", fontsize=16, weight='bold') # Set the main title.
    ax.set_xlabel("Principal Component 1")      # Set the x-axis label.
    ax.set_ylabel("Principal Component 2")      # Set the y-axis label.
    
    plt.tight_layout()                          # Adjust plot to prevent labels from overlapping.
    plt.savefig("visualization_stage_1_feature_space.png") # Save the final plot as a PNG file.
    print("--> Saved feature space plot to visualization_stage_1_feature_space.png")
    plt.close(fig)                              # Close the figure to free up memory.