import torch
import torch.nn as nn
import numpy as np
from functools import partial
import os
import mlflow

from qiskit_optimization import QuadraticProgram
from qiskit_algorithms.minimum_eigensolvers import SamplingVQE
from qiskit.circuit.library import QAOAAnsatz
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import SamplerV2
# --- Import the master translator ---
from qiskit import transpile

# --- Local Project Imports ---
from src.feature_engineering.data_setup import get_data_loaders
from src.feature_engineering.classical_components import Encoder
from src.feature_engineering.quantum_circuits import get_quantum_torch_layer
from .classical_classifier import SimpleClassifier, evaluate_model
from .qubo_formulation import create_qubo, binary_to_hyperparams

def generate_quantum_features(encoder, quantum_layer, data_loader, device):
    encoder.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            latent_vecs = encoder(images)
            q_features = quantum_layer(latent_vecs).cpu().numpy()
            all_features.append(q_features)
            all_labels.append(labels.numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)

def run_hyperparameter_tuning(config):
    print("\n--- MLOps Stage 2: Quantum-Accelerated Hyperparameter Tuning ---")
    cfg1 = config['stage_1_feature_engineering']
    cfg2 = config['stage_2_hyperparameter_tuning']
    device = torch.device("cpu")
    
    print("Loading feature extractor from Stage 1...")
    encoder = Encoder(cfg1['stage_1_latent_dim'], cfg1['stage_1_img_size'])
    encoder.load_state_dict(torch.load("saved_models/feature_extractor/hae_encoder.pth"))
    encoder.to(device)
    quantum_layer = get_quantum_torch_layer(cfg1['stage_1_latent_dim'])
    pqc_weights = np.load("saved_models/feature_extractor/hae_pqc_weights.npy")
    quantum_layer.weight = torch.nn.Parameter(torch.Tensor(pqc_weights))
    quantum_layer.to(device)
    
    print("Generating quantum-enhanced features for downstream task...")
    train_loader, _ = get_data_loaders(batch_size=128, n_samples=cfg2['stage_2_n_samples'], img_size=cfg1['stage_1_img_size'])
    features, labels = generate_quantum_features(encoder, quantum_layer, train_loader, device)
    print(f"Generated {features.shape[0]} feature vectors of dimension {cfg1['stage_1_latent_dim']}.")
    
    space = cfg2['hyperparameter_space']
    objective_function = partial(evaluate_model, quantum_features=features, labels=labels)
    qubo_matrix = create_qubo(objective_function, space)

    qp = QuadraticProgram()
    num_qubits = qubo_matrix.shape[0]
    for i in range(num_qubits):
        qp.binary_var(name=f'x{i}')
    qp.minimize(quadratic=qubo_matrix)
    hamiltonian, offset = qp.to_ising()

    print("\nConfiguring and running QAOA for HPO...")
    optimizer = COBYLA(maxiter=50)
    sampler = SamplerV2()
    
    ansatz_template = QAOAAnsatz(hamiltonian, reps=cfg2['stage_2_qaoa_reps'])
    
    # --- THE PERMANENT FIX ---
    # Use the powerful transpile function to break down all abstract gates
    # into the simple 'basis_gates' that the Aer simulator understands.
    decomposed_ansatz = transpile(ansatz_template, basis_gates=['u', 'cx'])
    
    qaoa_solver = SamplingVQE(sampler=sampler, ansatz=decomposed_ansatz, optimizer=optimizer)
    result = qaoa_solver.compute_minimum_eigenvalue(hamiltonian)
    best_solution_binary = result.best_measurement['bitstring']
    best_hyperparams = binary_to_hyperparams(best_solution_binary, space)
    
    print("\n--- QAOA HPO Results ---")
    print(f"Optimal binary string found by QAOA: {best_solution_binary}")
    print(f"Decoded optimal hyperparameters: {best_hyperparams}")
    final_accuracy = evaluate_model(best_hyperparams, features, labels)
    print(f"Final model accuracy with optimal hyperparameters: {final_accuracy:.4f}")

    mlflow.log_metric("stage_2_final_accuracy", final_accuracy)
    mlflow.log_params({f"stage_2_best_{key}": val for key, val in best_hyperparams.items()})
    
    print("\nTraining and saving final classifier with optimal hyperparameters...")
    final_model = SimpleClassifier(
        input_dim=features.shape[1],
        hidden_dim=int(best_hyperparams['hidden_dim']),
        dropout_rate=best_hyperparams['dropout']
    ).to(device)
    full_dataset = torch.utils.data.TensorDataset(torch.Tensor(features), torch.LongTensor(labels))
    train_loader_final = torch.utils.data.DataLoader(full_dataset, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer_final = torch.optim.Adam(final_model.parameters(), lr=best_hyperparams['lr'])
    final_model.train()
    for epoch in range(10):
        for data, target in train_loader_final:
            optimizer_final.zero_grad()
            output = final_model(data.to(device))
            loss = criterion(output, target.to(device))
            loss.backward()
            optimizer_final.step()
    
    save_dir = "saved_models/tuned_classifier"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "tuned_classifier.pth")
    torch.save(final_model.state_dict(), model_path)
    print(f"Final tuned classifier saved to: {model_path}")
    print("Logging tuned classifier artifact to MLflow...")
    mlflow.log_artifact(model_path, artifact_path="stage_2_tuned_classifier")
    print("Logging tuned classifier artifact to MLflow...")
    mlflow.log_artifact(model_path, artifact_path="stage_2_tuned_classifier")
    print("\n--- Hyperparameter Tuning Stage Complete ---")