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

# The function now accepts the master config object
def create_feature_space_plot(config):
    print("\n[VISUALIZATION] Generating plot for Stage 1: Feature Space...")
    
    # --- Read new, specific keys from the config file ---
    cfg1 = config['stage_1_feature_engineering']
    device = torch.device("cpu")
    
    # Load the saved artifacts
    encoder = Encoder(cfg1['stage_1_latent_dim'], cfg1['stage_1_img_size'])
    encoder.load_state_dict(torch.load("saved_models/feature_extractor/hae_encoder.pth"))
    encoder.to(device)
    
    quantum_layer = get_quantum_torch_layer(cfg1['stage_1_latent_dim'])
    pqc_weights = np.load("saved_models/feature_extractor/hae_pqc_weights.npy")
    quantum_layer.weight = torch.nn.Parameter(torch.Tensor(pqc_weights))
    quantum_layer.to(device)

    # Generate features to plot using config parameters
    vis_loader, _ = get_data_loaders(
        batch_size=cfg1['stage_1_n_samples'], 
        n_samples=cfg1['stage_1_n_samples'], 
        img_size=cfg1['stage_1_img_size']
    )
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