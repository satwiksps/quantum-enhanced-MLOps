import torch                                    # The main PyTorch library.
import numpy as np                              # For numerical operations and creating noisy data.
import os                                       # For handling file paths.
import mlflow                                   # For logging metrics from the monitoring stage.

from qiskit.circuit.library import ZFeatureMap    # A standard circuit for encoding classical data into a quantum state.
from qiskit_aer.primitives import SamplerV2       # The fast, local quantum simulator.
from qiskit_machine_learning.kernels import FidelityQuantumKernel # A method to calculate the "similarity" between quantum states.
from sklearn.svm import OneClassSVM             # The classical SVM algorithm used for anomaly detection.

# --- Local Project Imports ---
from src.feature_engineering.data_setup import get_data_loaders
from src.feature_engineering.classical_components import Encoder
from src.feature_engineering.quantum_circuits import get_quantum_torch_layer
from src.hyperparameter_tuning.tune_with_qaoa import generate_quantum_features

# The function now accepts the master config object
def run_drift_detection(config):
    print("\n--- MLOps Stage 3: Quantum-Enhanced Production Monitoring ---")
    
    # --- Read parameters from the config file ---
    cfg1 = config['stage_1_feature_engineering']
    cfg3 = config['stage_3_production_monitoring']
    device = torch.device("cpu")                # Set the device to CPU.
    
    # --- Load the trained feature extractor from Stage 1 ---
    print("Loading feature extractor from Stage 1...")
    encoder = Encoder(cfg1['stage_1_latent_dim'], cfg1['stage_1_img_size'])
    encoder.load_state_dict(torch.load("saved_models/feature_extractor/hae_encoder.pth")) # Load saved encoder weights.
    encoder.to(device)
    quantum_layer = get_quantum_torch_layer(cfg1['stage_1_latent_dim'])
    pqc_weights = np.load("saved_models/feature_extractor/hae_pqc_weights.npy") # Load saved PQC weights.
    quantum_layer.weight = torch.nn.Parameter(torch.Tensor(pqc_weights))
    quantum_layer.to(device)

    # --- Generate a dataset of "normal" data to train the monitor ---
    print("Generating quantum features from 'normal' production data to train the monitor...")
    train_loader, _ = get_data_loaders(
        batch_size=cfg3['stage_3_n_samples'], 
        n_samples=cfg3['stage_3_n_samples'], 
        img_size=cfg1['stage_1_img_size']
    )
    normal_features, _ = generate_quantum_features(encoder, quantum_layer, train_loader, device)
    
    # --- Configure the One-Class Quantum SVM ---
    print("Configuring One-Class QSVM for drift detection...")
    feature_map = ZFeatureMap(feature_dimension=cfg1['stage_1_latent_dim'], reps=2) # The circuit to encode data.
    sampler = SamplerV2()                           # Initialize the local simulator.
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map) # Define the quantum kernel.
    quantum_kernel.sampler = sampler                # Assign the simulator to the kernel.
    
    # Initialize scikit-learn's OneClassSVM, but tell it to use our quantum kernel as the similarity function.
    qsvm_monitor = OneClassSVM(kernel=quantum_kernel.evaluate, nu=cfg3['stage_3_nu_param'])

    # --- Train the monitor on only the "normal" data ---
    print("Training the QSVM monitor...")
    qsvm_monitor.fit(normal_features)               # The SVM learns the boundary of the normal data.
    print("QSVM monitor training complete.")
    
    # --- Simulate a live data stream containing both normal and drifted data ---
    print("\nSimulating a production data stream with potential data drift...")
    noise = np.random.normal(0, 0.8, normal_features.shape) # Create some random noise.
    anomalous_features = normal_features + noise    # Create "drifted" data by adding noise.
    production_stream = np.concatenate([normal_features[:10], anomalous_features[:10]]) # Create a small test stream.
    
    # --- Use the trained monitor to make predictions on the new data ---
    predictions = qsvm_monitor.predict(production_stream)
    
    # --- Log the results of the monitoring test to MLflow ---
    num_anomalies_detected = np.sum(predictions == -1) # Count how many data points were flagged as anomalous.
    mlflow.log_metric("stage_3_anomalies_detected_in_stream", num_anomalies_detected) # Log this count to MLflow.
    print(f"Logged metric to MLflow: Detected {num_anomalies_detected} anomalies in the test stream.")
    
    # --- Display the results in the terminal ---
    print("\n--- Data Drift Detection Results ---")
    print("Prediction key: 1 = Inlier (Normal), -1 = Outlier (Anomaly/Drift)")
    for i, p in enumerate(predictions):
        data_type = "Normal" if i < 10 else "Anomalous" # Check if it was a normal or anomalous point.
        status = "Normal" if p == 1 else "ANOMALY DETECTED" # Check the SVM's prediction.
        print(f"Data point {i+1} (True type: {data_type}) -> Prediction: {status}")

    print("\n--- Production Monitoring Stage Complete ---")