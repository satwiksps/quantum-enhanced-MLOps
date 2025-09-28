import torch
import numpy as np
import os

from qiskit.circuit.library import ZFeatureMap
from qiskit_aer.primitives import SamplerV2
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import OneClassSVM   # ✅ Correct import for one-class SVM

from src.feature_engineering.data_setup import get_data_loaders
from src.feature_engineering.classical_components import Encoder
from src.feature_engineering.quantum_circuits import get_quantum_torch_layer
from src.hyperparameter_tuning.tune_with_qaoa import generate_quantum_features


def run_drift_detection():
    """
    Trains and demonstrates the One-Class Quantum SVM for production monitoring and data drift detection.
    """
    print("\n--- MLOps Stage 3: Quantum-Enhanced Production Monitoring ---")
    
    device = torch.device("cpu")
    latent_dim, img_size = 4, 14

    print("Loading feature extractor from Stage 1...")
    encoder = Encoder(latent_dim, img_size)
    encoder.load_state_dict(torch.load("saved_models/feature_extractor/hae_encoder.pth"))
    encoder.to(device)
    
    quantum_layer = get_quantum_torch_layer(latent_dim)
    pqc_weights = np.load("saved_models/feature_extractor/hae_pqc_weights.npy")
    quantum_layer.weight = torch.nn.Parameter(torch.Tensor(pqc_weights))
    quantum_layer.to(device)

    print("Generating quantum features from 'normal' production data to train the monitor...")
    train_loader, _ = get_data_loaders(batch_size=64, n_samples=200, img_size=img_size)
    normal_features, _ = generate_quantum_features(encoder, quantum_layer, train_loader, device)
    
    print("Configuring One-Class QSVM for drift detection...")
    feature_map = ZFeatureMap(feature_dimension=latent_dim, reps=2)
    
    # Modern kernel setup
    sampler = SamplerV2()
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    quantum_kernel.sampler = sampler

    # ✅ Use OneClassSVM instead of QSVC
    qsvm_monitor = OneClassSVM(kernel=quantum_kernel.evaluate, nu=0.1)

    print("Training the QSVM monitor...")
    qsvm_monitor.fit(normal_features)
    print("QSVM monitor training complete.")
    
    print("\nSimulating a production data stream with potential data drift...")
    noise = np.random.normal(0, 0.8, normal_features.shape)
    anomalous_features = normal_features + noise
    production_stream = np.concatenate([normal_features[:10], anomalous_features[:10]])
    
    predictions = qsvm_monitor.predict(production_stream)
    
    print("\n--- Data Drift Detection Results ---")
    print("Prediction key: 1 = Inlier (Normal), -1 = Outlier (Anomaly/Drift)")
    for i, p in enumerate(predictions):
        data_type = "Normal" if i < 10 else "Anomalous"
        status = "Normal" if p == 1 else "ANOMALY DETECTED"
        print(f"Data point {i+1} (True type: {data_type}) -> Prediction: {status}")

    print("\n--- Production Monitoring Stage Complete ---")
