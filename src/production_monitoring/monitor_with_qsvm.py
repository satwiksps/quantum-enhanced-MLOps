import torch                                    # The main PyTorch library.
import numpy as np                              # For numerical operations.
import os                                       # For handling file paths.
import mlflow                                   # For logging metrics.
import sys                                      # Used here to interact with the system (for the trigger).

from qiskit.circuit.library import ZFeatureMap    # A standard circuit for encoding classical data.
from qiskit_aer.primitives import SamplerV2       # The fast, local quantum simulator.
from qiskit_machine_learning.kernels import FidelityQuantumKernel # A method to calculate "quantum similarity".
from sklearn.svm import OneClassSVM             # The classical SVM algorithm used for anomaly detection.

# --- Local Project Imports ---
from src.feature_engineering.data_setup import get_data_loaders
from src.feature_engineering.classical_components import Encoder
from src.feature_engineering.quantum_circuits import get_quantum_torch_layer
from src.hyperparameter_tuning.tune_with_qaoa import generate_quantum_features

def run_drift_detection(config):
    print("\n--- MLOps Stage 3: Quantum-Enhanced Production Monitoring ---")
    cfg1 = config['stage_1_feature_engineering'] # Get Stage 1 parameters from the config.
    cfg3 = config['stage_3_production_monitoring'] # Get Stage 3 parameters from the config.
    device = torch.device("cpu")                # Set the device to CPU.
    
    print("Loading feature extractor from Stage 1...")
    encoder = Encoder(cfg1['stage_1_latent_dim'], cfg1['stage_1_img_size']) # Initialize the encoder architecture.
    encoder.load_state_dict(torch.load("saved_models/feature_extractor/hae_encoder.pth")) # Load its saved weights.
    encoder.to(device)
    quantum_layer = get_quantum_torch_layer(cfg1['stage_1_latent_dim']) # Initialize the quantum layer.
    pqc_weights = np.load("saved_models/feature_extractor/hae_pqc_weights.npy") # Load its saved weights.
    quantum_layer.weight = torch.nn.Parameter(torch.Tensor(pqc_weights)) # Assign the weights.
    quantum_layer.to(device)

    print("Generating quantum features from 'normal' production data to train the monitor...")
    train_loader, _ = get_data_loaders(
        batch_size=cfg3['stage_3_n_samples'], 
        n_samples=cfg3['stage_3_n_samples'], 
        img_size=cfg1['stage_1_img_size']
    )
    normal_features, _ = generate_quantum_features(encoder, quantum_layer, train_loader, device)
    
    # This line was missing in the original file but is crucial. It configures the quantum kernel.
    quantum_kernel = FidelityQuantumKernel(feature_map=ZFeatureMap(feature_dimension=cfg1['stage_1_latent_dim'], reps=2))
    quantum_kernel.sampler = SamplerV2()
    
    qsvm_monitor = OneClassSVM(kernel=quantum_kernel.evaluate, nu=cfg3['stage_3_nu_param']) # Initialize the SVM with the quantum kernel.

    print("Training the QSVM monitor...")
    qsvm_monitor.fit(normal_features)               # Train the monitor on only "good" data.
    print("QSVM monitor training complete.")
    
    print("\nSimulating a production data stream with potential data drift...")
    noise = np.random.normal(0, 0.8, normal_features.shape) # Create some random noise.
    anomalous_features = normal_features + noise    # Create "drifted" data by adding noise.
    
    # --- Drift Detection Logic ---
    production_stream = np.concatenate([normal_features, anomalous_features]) # Combine good and bad data for testing.
    predictions = qsvm_monitor.predict(production_stream) # Get the monitor's predictions.
    
    true_labels = np.array([1]*len(normal_features) + [-1]*len(anomalous_features)) # Create ground truth labels.
    
    anomalies_missed = np.sum((predictions == 1) & (true_labels == -1)) # Count how many anomalies were missed.
    total_anomalies = len(anomalous_features)       # Get the total number of anomalies.
    drift_rate = anomalies_missed / total_anomalies if total_anomalies > 0 else 0 # Calculate the miss rate.
    
    print(f"\nDrift Analysis: The monitor missed {anomalies_missed} out of {total_anomalies} anomalous data points.")
    print(f"Calculated Drift Rate: {drift_rate:.2%}")
    mlflow.log_metric("stage_3_drift_rate", drift_rate) # Log the result to MLflow.

    # The threshold to decide if retraining is needed.
    drift_threshold = 0.5 # This can be adjusted based on acceptable risk levels.
    
    # Create a simple text file to signal the status to the CI/CD pipeline.
    with open("drift_status.txt", "w") as f:
        if drift_rate > drift_threshold:            # Check if the miss rate is too high.
            print(f"[ALERT] Drift rate ({drift_rate:.2%}) exceeds threshold ({drift_threshold:.2%}). Signaling for retraining.")
            f.write("DRIFT_DETECTED")               # Write the "emergency" signal.
        else:
            print(f"Drift rate ({drift_rate:.2%}) is within acceptable limits.")
            f.write("NO_DRIFT")                     # Write the "all clear" signal.
            
    print("\n--- Production Monitoring Stage Complete ---")