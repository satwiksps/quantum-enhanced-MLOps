# --- Visualization for Stage 3: FAST VERSION ---
# This script generates a simplified plot quickly by using fewer data points
# and not rendering the full decision boundary.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM

from src.feature_engineering.data_setup import get_data_loaders
from src.feature_engineering.classical_components import Encoder
from src.feature_engineering.quantum_circuits import get_quantum_torch_layer
from src.hyperparameter_tuning.tune_with_qaoa import generate_quantum_features
from qiskit.circuit.library import ZFeatureMap
from qiskit_aer.primitives import SamplerV2
from qiskit_machine_learning.kernels import FidelityQuantumKernel

colors = {'normal': '#0077B6', 'anomalous': '#F4A261', 'support': '#E76F51'}

def create_drift_detection_plot_fast():
    """
    Generates a fast version of the drift detection plot.
    """
    print("\n[VISUALIZATION] Generating FAST plot for Stage 3: Drift Detection...")

    device = torch.device("cpu")
    latent_dim, img_size = 4, 14
    
    # --- SPEED INCREASE 1: Use a much smaller dataset for the plot ---
    N_SAMPLES_FOR_PLOT = 50  # Reduced from 200
    print(f"Using a small sample size ({N_SAMPLES_FOR_PLOT}) for fast plotting...")

    # Load Stage 1 artifacts
    encoder = Encoder(latent_dim, img_size)
    encoder.load_state_dict(torch.load("saved_models/feature_extractor/hae_encoder.pth"))
    quantum_layer = get_quantum_torch_layer(latent_dim)
    pqc_weights = np.load("saved_models/feature_extractor/hae_pqc_weights.npy")
    quantum_layer.weight = torch.nn.Parameter(torch.Tensor(pqc_weights))
    
    # Set up the Quantum SVM
    feature_map = ZFeatureMap(feature_dimension=latent_dim, reps=2)
    sampler = SamplerV2()
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    quantum_kernel.sampler = sampler
    qsvm_monitor = OneClassSVM(kernel=quantum_kernel.evaluate, nu=0.1)
    
    # Generate a small set of normal and anomalous data
    data_loader, _ = get_data_loaders(batch_size=N_SAMPLES_FOR_PLOT, n_samples=N_SAMPLES_FOR_PLOT, img_size=img_size)
    normal_features, _ = generate_quantum_features(encoder, quantum_layer, data_loader, device)
    noise = np.random.normal(0, 0.8, normal_features.shape)
    anomalous_features = normal_features + noise
    
    print("Training temporary QSVM for visualization (this will be much faster)...")
    qsvm_monitor.fit(normal_features)
    
    all_features = np.concatenate([normal_features, anomalous_features])
    
    # --- SPEED INCREASE 2: Do not plot the background boundary ---
    # We will just plot the points themselves.
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(all_features)
    normal_2d = features_2d[:len(normal_features)]
    anomalous_2d = features_2d[len(normal_features):]

    # Get the support vectors - the points the SVM thinks are most important
    support_vectors_2d = pca.transform(qsvm_monitor.support_vectors_)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the data points
    ax.scatter(normal_2d[:, 0], normal_2d[:, 1], c=colors['normal'], label='Normal Data', s=50)
    ax.scatter(anomalous_2d[:, 0], anomalous_2d[:, 1], c=colors['anomalous'], label='Anomalous Data (Drift)', s=50)
    
    # Highlight the support vectors to show what the SVM learned
    ax.scatter(support_vectors_2d[:, 0], support_vectors_2d[:, 1], 
               s=150, facecolors='none', edgecolors=colors['support'], linewidth=2.5,
               label='Support Vectors (Boundary Definers)')

    ax.set_title("Quantum SVM Monitoring for Data Drift (Fast Viz)", fontsize=16, weight='bold')
    ax.set_xlabel("Latent Space PC1")
    ax.set_ylabel("Latent Space PC2")
    ax.legend()
    plt.tight_layout()
    plt.savefig("visualization_stage_3_drift_FAST.png")
    print("--> Saved FAST drift detection plot to visualization_stage_3_drift_FAST.png")
    plt.close(fig)