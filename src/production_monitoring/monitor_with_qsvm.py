import torch
import numpy as np
import os
import mlflow

from qiskit.circuit.library import ZFeatureMap
from qiskit_aer.primitives import SamplerV2
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import OneClassSVM

from src.feature_engineering.data_setup import get_data_loaders
from src.feature_engineering.classical_components import Encoder
from src.feature_engineering.quantum_circuits import get_quantum_torch_layer
from src.hyperparameter_tuning.tune_with_qaoa import generate_quantum_features

def run_drift_detection(config):
    print("\n--- MLOps Stage 3: Quantum-Enhanced Production Monitoring ---")
    cfg1 = config['stage_1_feature_engineering']
    cfg3 = config['stage_3_production_monitoring']
    device = torch.device("cpu")
    
    print("Loading feature extractor from Stage 1...")
    encoder = Encoder(cfg1['stage_1_latent_dim'], cfg1['stage_1_img_size'])
    encoder.load_state_dict(torch.load("saved_models/feature_extractor/hae_encoder.pth"))
    encoder.to(device)
    quantum_layer = get_quantum_torch_layer(cfg1['stage_1_latent_dim'])
    pqc_weights = np.load("saved_models/feature_extractor/hae_pqc_weights.npy")
    quantum_layer.weight = torch.nn.Parameter(torch.Tensor(pqc_weights))
    quantum_layer.to(device)

    print("Generating quantum features from 'normal' production data to train the monitor...")
    train_loader, _ = get_data_loaders(
        batch_size=cfg3['stage_3_n_samples'], 
        n_samples=cfg3['stage_3_n_samples'], 
        img_size=cfg1['stage_1_img_size']
    )
    normal_features, _ = generate_quantum_features(encoder, quantum_layer, train_loader, device)
    
    print("Configuring One-Class QSVM for drift detection...")
    feature_map = ZFeatureMap(feature_dimension=cfg1['stage_1_latent_dim'], reps=2)
    sampler = SamplerV2()
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    quantum_kernel.sampler = sampler
    qsvm_monitor = OneClassSVM(kernel=quantum_kernel.evaluate, nu=cfg3['stage_3_nu_param'])

    print("Training the QSVM monitor...")
    qsvm_monitor.fit(normal_features)
    print("QSVM monitor training complete.")
    
    print("\nSimulating a production data stream with potential data drift...")
    noise = np.random.normal(0, 0.8, normal_features.shape)
    anomalous_features = normal_features + noise
    production_stream = np.concatenate([normal_features[:10], anomalous_features[:10]])
    predictions = qsvm_monitor.predict(production_stream)
    
    num_anomalies_detected = np.sum(predictions == -1)
    mlflow.log_metric("stage_3_anomalies_detected_in_stream", num_anomalies_detected)
    print(f"Logged metric to MLflow: Detected {num_anomalies_detected} anomalies in the test stream.")
    
    print("\n--- Data Drift Detection Results ---")
    print("Prediction key: 1 = Inlier (Normal), -1 = Outlier (Anomaly/Drift)")
    for i, p in enumerate(predictions):
        data_type = "Normal" if i < 10 else "Anomalous"
        status = "Normal" if p == 1 else "ANOMALY DETECTED"
        print(f"Data point {i+1} (True type: {data_type}) -> Prediction: {status}")

    print("\n--- Production Monitoring Stage Complete ---")