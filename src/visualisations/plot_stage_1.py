# --- Visualization for Stage 1: Feature Space ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.decomposition import PCA

from src.feature_engineering.data_setup import get_data_loaders
from src.feature_engineering.classical_components import Encoder
from src.feature_engineering.quantum_circuits import get_quantum_torch_layer
from src.hyperparameter_tuning.tune_with_qaoa import generate_quantum_features

def create_feature_space_plot():
    """
    Loads the trained feature extractor and generates a 2D PCA plot of the
    quantum-native feature space, colored by digit class.
    """
    print("\n[VISUALIZATION] Generating plot for Stage 1: Feature Space...")
    
    device = torch.device("cpu")
    latent_dim, img_size = 4, 14

    # Load the saved artifacts from the main pipeline run
    encoder = Encoder(latent_dim, img_size)
    encoder.load_state_dict(torch.load("saved_models/feature_extractor/hae_encoder.pth"))
    encoder.to(device)
    
    quantum_layer = get_quantum_torch_layer(latent_dim)
    pqc_weights = np.load("saved_models/feature_extractor/hae_pqc_weights.npy")
    quantum_layer.weight = torch.nn.Parameter(torch.Tensor(pqc_weights))
    quantum_layer.to(device)

    # Generate features to plot
    vis_loader, _ = get_data_loaders(batch_size=400, n_samples=400, img_size=img_size)
    features, labels = generate_quantum_features(encoder, quantum_layer, vis_loader, device)
    
    # Create the plot
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    legend1 = ax.legend(*scatter.legend_elements(), title="Digits")
    ax.add_artist(legend1)
    ax.set_title("Quantum-Native Feature Space", fontsize=16, weight='bold')
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    
    plt.tight_layout()
    plt.savefig("visualization_stage_1_feature_space.png")
    print("--> Saved feature space plot to visualization_stage_1_feature_space.png")
    plt.close(fig)

if __name__ == '__main__':
    create_feature_space_plot()