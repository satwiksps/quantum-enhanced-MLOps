# --- Visualization for Stage 3: Drift Detection ---
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

colors = {'normal': '#0077B6', 'anomalous': '#F4A261'}

def create_drift_detection_plots():
    """
    Loads artifacts, re-trains the QSVM, and generates the decision boundary
    plot and a confusion matrix to visualize drift detection performance.
    """
    print("\n[VISUALIZATION] Generating plots for Stage 3: Drift Detection...")

    device = torch.device("cpu")
    latent_dim, img_size = 4, 14
    
    # Load Stage 1 artifacts
    encoder = Encoder(latent_dim, img_size)
    encoder.load_state_dict(torch.load("saved_models/feature_extractor/hae_encoder.pth"))
    quantum_layer = get_quantum_torch_layer(latent_dim)
    pqc_weights = np.load("saved_models/feature_extractor/hae_pqc_weights.npy")
    quantum_layer.weight = torch.nn.Parameter(torch.Tensor(pqc_weights))
    
    # We must re-train the QSVM here to create the plot
    feature_map = ZFeatureMap(feature_dimension=latent_dim, reps=2)
    sampler = SamplerV2()
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    quantum_kernel.sampler = sampler
    qsvm_monitor = OneClassSVM(kernel=quantum_kernel.evaluate, nu=0.1)
    
    data_loader, _ = get_data_loaders(batch_size=200, n_samples=200, img_size=img_size)
    normal_features, _ = generate_quantum_features(encoder, quantum_layer, data_loader, device)
    noise = np.random.normal(0, 0.8, normal_features.shape)
    anomalous_features = normal_features + noise
    
    print("Training temporary QSVM for visualization...")
    qsvm_monitor.fit(normal_features)
    
    all_features = np.concatenate([normal_features, anomalous_features])
    all_predictions = qsvm_monitor.predict(all_features)
    
    # Create the Decision Boundary Plot
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(all_features)
    normal_2d = features_2d[:len(normal_features)]
    anomalous_2d = features_2d[len(normal_features):]

    fig, ax = plt.subplots(figsize=(10, 8))
    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = qsvm_monitor.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    ax.scatter(normal_2d[:, 0], normal_2d[:, 1], c=colors['normal'], label='Normal Data', s=50, edgecolor='k')
    ax.scatter(anomalous_2d[:, 0], anomalous_2d[:, 1], c=colors['anomalous'], label='Anomalous Data (Drift)', s=50, edgecolor='k')
    ax.set_title("Quantum SVM Monitoring for Data Drift", fontsize=16, weight='bold')
    ax.set_xlabel("Latent Space PC1")
    ax.set_ylabel("Latent Space PC2")
    ax.legend()
    plt.tight_layout()
    plt.savefig("visualization_stage_3_drift_boundary.png")
    print("--> Saved drift detection plot to visualization_stage_3_drift_boundary.png")
    plt.close(fig)

    # Create the Confusion Matrix Plot
    true_labels = np.array([1]*len(normal_features) + [-1]*len(anomalous_features))
    cm = confusion_matrix(true_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Anomalous', 'Normal'])
    disp.plot(cmap='Blues')
    plt.title("Drift Detection Confusion Matrix")
    plt.savefig("visualization_stage_3_confusion_matrix.png")
    print("--> Saved confusion matrix to visualization_stage_3_confusion_matrix.png")
    plt.close()

if __name__ == '__main__':
    create_drift_detection_plots()