import torch
import torch.nn as nn
import numpy as np
from functools import partial
import os

from qiskit_optimization import QuadraticProgram
from qiskit_algorithms.minimum_eigensolvers import SamplingVQE
from qiskit.circuit.library import QAOAAnsatz
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import SamplerV2
from qiskit import transpile

# --- Local Project Imports ---
from src.feature_engineering.data_setup import get_data_loaders
from src.feature_engineering.classical_components import Encoder
from src.feature_engineering.quantum_circuits import get_quantum_torch_layer
from .classical_classifier import SimpleClassifier, evaluate_model
from .qubo_formulation import define_hyperparameter_space, create_qubo, binary_to_hyperparams


def generate_quantum_features(encoder, quantum_layer, data_loader, device):
    """Uses the trained HAE to generate features for the downstream task."""
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


def run_hyperparameter_tuning():
    print("\n--- MLOps Stage 2: Quantum-Accelerated Hyperparameter Tuning ---")
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
    
    print("Generating quantum-enhanced features for downstream task...")
    train_loader, _ = get_data_loaders(batch_size=64, n_samples=1000, img_size=img_size)
    features, labels = generate_quantum_features(encoder, quantum_layer, train_loader, device)
    print(f"Generated {features.shape[0]} feature vectors of dimension {features.shape[1]}.")
    
    # Build QUBO problem
    space = define_hyperparameter_space()
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
    
    # --- FIX: transpile QAOA ansatz fully into Aer-supported gates ---
    ansatz_template = QAOAAnsatz(hamiltonian, reps=2)
    decomposed_ansatz = transpile(
        ansatz_template,
        basis_gates=["cx", "rz", "ry", "rx"],
        optimization_level=3
    )

    print("Circuit operations after transpilation:", decomposed_ansatz.count_ops())
    
    qaoa_solver = SamplingVQE(
        sampler=sampler,
        ansatz=decomposed_ansatz,
        optimizer=optimizer
    )
    
    result = qaoa_solver.compute_minimum_eigenvalue(hamiltonian)
    
    best_solution_binary = result.best_measurement['bitstring']
    best_hyperparams = binary_to_hyperparams(best_solution_binary, space)
    
    print("\n--- QAOA HPO Results ---")
    print(f"Optimal binary string found by QAOA: {best_solution_binary}")
    print(f"Decoded optimal hyperparameters: {best_hyperparams}")

    final_accuracy = evaluate_model(best_hyperparams, features, labels)
    print(f"Final model accuracy with optimal hyperparameters: {final_accuracy:.4f}")
    # --- NEW SECTION: "BAKE AND SAVE THE FINAL CAKE" ---
    print("\nTraining and saving final classifier with optimal hyperparameters...")
    
    # 1. Create the final model with the winning recipe
    final_model = SimpleClassifier(
        input_dim=features.shape[1],
        hidden_dim=int(best_hyperparams['hidden_dim']),
        dropout_rate=best_hyperparams['dropout']
    ).to(device)

    # 2. Prepare for training on the full dataset
    full_dataset = torch.utils.data.TensorDataset(torch.Tensor(features), torch.LongTensor(labels))
    train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_hyperparams['lr'])

    # 3. Train the model for a few epochs
    final_model.train()
    for epoch in range(10): # Train for 10 epochs
        for data, target in train_loader:
            optimizer.zero_grad()
            output = final_model(data.to(device))
            loss = criterion(output, target.to(device))
            loss.backward()
            optimizer.step()
    
    # 4. Save the final, trained model
    save_dir = "saved_models/tuned_classifier"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "tuned_classifier.pth")
    torch.save(final_model.state_dict(), model_path)
    print(f"Final tuned classifier saved to: {model_path}")
    # --- END OF NEW SECTION ---
    print("\n--- Hyperparameter Tuning Stage Complete ---")
    return best_hyperparams, final_accuracy