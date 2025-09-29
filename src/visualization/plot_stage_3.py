import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import OneClassSVM

from src.feature_engineering.data_setup import get_data_loaders
from src.feature_engineering.classical_components import Encoder
from src.feature_engineering.quantum_circuits import get_quantum_torch_layer
from src.hyperparameter_tuning.tune_with_qaoa import generate_quantum_features
from qiskit.circuit.library import ZFeatureMap
from qiskit_aer.primitives import SamplerV2
from qiskit_machine_learning.kernels import FidelityQuantumKernel

colors = {'normal': '#0077B6', 'anomalous': '#F4A261', 'support': '#E76F51'}

# The function now accepts the master config object
def create_drift_detection_plots(config):
    # --- Read new, specific keys from the config file ---
    cfg1 = config['stage_1_feature_engineering']
    cfg3 = config['stage_3_production_monitoring']
    mode = cfg3['visualization_mode']
    
    print(f"\n[VISUALIZATION] Generating Stage 3 plots in '{mode}' mode...")

    device = torch.device("cpu")
    latent_dim = cfg1['stage_1_latent_dim']
    img_size = cfg1['stage_1_img_size']
    
    # --- Common Setup for Both Modes ---
    encoder = Encoder(latent_dim, img_size)
    encoder.load_state_dict(torch.load("saved_models/feature_extractor/hae_encoder.pth"))
    quantum_layer = get_quantum_torch_layer(latent_dim)
    pqc_weights = np.load("saved_models/feature_extractor/hae_pqc_weights.npy")
    quantum_layer.weight = torch.nn.Parameter(torch.Tensor(pqc_weights))
    
    feature_map = ZFeatureMap(feature_dimension=latent_dim, reps=2)
    sampler = SamplerV2()
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    quantum_kernel.sampler = sampler
    qsvm_monitor = OneClassSVM(kernel=quantum_kernel.evaluate, nu=cfg3['stage_3_nu_param'])
    
    # --- Logic Switch Based on Config ---
    if mode == 'fast':
        # --- FAST MODE LOGIC ---
        n_samples = cfg3['stage_3_fast_plot_n_samples']
        print(f"Using a small sample size ({n_samples}) for fast plotting...")
        
        data_loader, _ = get_data_loaders(batch_size=n_samples, n_samples=n_samples, img_size=img_size)
        normal_features, _ = generate_quantum_features(encoder, quantum_layer, data_loader, device)
        
        print("Training temporary QSVM for fast visualization...")
        qsvm_monitor.fit(normal_features)
        
        noise = np.random.normal(0, 0.8, normal_features.shape)
        anomalous_features = normal_features + noise
        all_features = np.concatenate([normal_features, anomalous_features])
        
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(all_features)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(features_2d[:n_samples, 0], features_2d[:n_samples, 1], c=colors['normal'], label='Normal Data', s=50)
        ax.scatter(features_2d[n_samples:, 0], features_2d[n_samples:, 1], c=colors['anomalous'], label='Anomalous Data', s=50)
        
        support_indices = qsvm_monitor.support_
        support_vectors = normal_features[support_indices]
        support_vectors_2d = pca.transform(support_vectors)
        ax.scatter(support_vectors_2d[:, 0], support_vectors_2d[:, 1], s=150, facecolors='none', edgecolors=colors['support'], linewidth=2.5, label='Support Vectors')

        ax.set_title("Quantum SVM Monitoring (Fast Visualization)", fontsize=16, weight='bold')
        ax.legend()
        plt.savefig("visualization_stage_3_drift_FAST.png")
        print("--> Saved FAST drift detection plot.")
        plt.close(fig)

    else:
        # --- HIGH-QUALITY (SLOW) MODE LOGIC ---
        n_samples = cfg3['stage_3_n_samples']
        print(f"Using sample size ({n_samples}) for high-quality plotting...")
        
        data_loader, _ = get_data_loaders(batch_size=n_samples, n_samples=n_samples, img_size=img_size)
        normal_features, _ = generate_quantum_features(encoder, quantum_layer, data_loader, device)
        
        print("Training temporary QSVM for visualization... (This will take several minutes)")
        qsvm_monitor.fit(normal_features)

        noise = np.random.normal(0, 0.8, normal_features.shape)
        anomalous_features = normal_features + noise
        all_features = np.concatenate([normal_features, anomalous_features])
        
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(all_features)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
        y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        
        print("Plotting decision boundary... (This is the slowest part)")
        Z = qsvm_monitor.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
        ax.scatter(features_2d[:n_samples, 0], features_2d[:n_samples, 1], c=colors['normal'], label='Normal Data', s=50, edgecolor='k')
        ax.scatter(features_2d[n_samples:, 0], features_2d[n_samples:, 1], c=colors['anomalous'], label='Anomalous Data', s=50, edgecolor='k')
        ax.set_title("Quantum SVM Monitoring for Data Drift", fontsize=16, weight='bold')
        ax.legend()
        plt.savefig("visualization_stage_3_drift_boundary.png")
        print("--> Saved high-quality drift detection plot.")
        plt.close(fig)

    # --- Confusion Matrix (for both modes) ---
    all_predictions = qsvm_monitor.predict(all_features)
    true_labels = np.array([1]*len(normal_features) + [-1]*len(anomalous_features))
    cm = confusion_matrix(true_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Anomalous', 'Normal'])
    disp.plot(cmap='Blues')
    plt.title(f"Drift Detection Confusion Matrix ({mode.capitalize()} Mode)")
    plt.savefig(f"visualization_stage_3_confusion_matrix_{mode.upper()}.png")
    print(f"--> Saved confusion matrix for {mode} mode.")
    plt.close()