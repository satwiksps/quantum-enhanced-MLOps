import torch                                    # The main PyTorch library.
import torch.nn as nn                           # Provides neural network layers.
import numpy as np                              # For numerical operations.
from functools import partial                   # A helper to pre-fill function arguments.
import os                                       # For handling file paths and directories.
import mlflow                                   # For logging metrics and artifacts.

from qiskit_optimization import QuadraticProgram  # The object for defining optimization problems.
from qiskit_algorithms.minimum_eigensolvers import SamplingVQE # The modern solver for variational algorithms.
from qiskit.circuit.library import QAOAAnsatz   # The specific circuit blueprint for QAOA.
from qiskit_algorithms.optimizers import COBYLA # A classical optimizer used within the quantum algorithm.
from qiskit_aer.primitives import SamplerV2       # The fast, local quantum simulator.
from qiskit import transpile                    # The powerful "master translator" for quantum circuits.

# --- Local Project Imports ---
from src.feature_engineering.data_setup import get_data_loaders
from src.feature_engineering.classical_components import Encoder
from src.feature_engineering.quantum_circuits import get_quantum_torch_layer
from .classical_classifier import SimpleClassifier, evaluate_model
from .qubo_formulation import create_qubo, binary_to_hyperparams

def generate_quantum_features(encoder, quantum_layer, data_loader, device):
    encoder.eval()                              # Set the encoder to evaluation mode.
    all_features, all_labels = [], []           # Initialize lists to store results.
    with torch.no_grad():                       # Disable gradient calculation for speed.
        for images, labels in data_loader:
            images = images.to(device)          # Move images to the correct device.
            latent_vecs = encoder(images)       # Use the encoder to create classical latent vectors.
            q_features = quantum_layer(latent_vecs).cpu().numpy() # Process vectors through the PQC to get quantum features.
            all_features.append(q_features)     # Collect the feature batches.
            all_labels.append(labels.numpy())   # Collect the label batches.
    return np.concatenate(all_features), np.concatenate(all_labels) # Combine all batches into single arrays.

def run_hyperparameter_tuning(config):
    print("\n--- MLOps Stage 2: Quantum-Accelerated Hyperparameter Tuning ---")
    cfg1 = config['stage_1_feature_engineering'] # Get Stage 1 parameters from the config.
    cfg2 = config['stage_2_hyperparameter_tuning'] # Get Stage 2 parameters from the config.
    device = torch.device("cpu")                # Set the device to CPU.
    
    # --- Load the trained feature extractor from Stage 1 ---
    print("Loading feature extractor from Stage 1...")
    encoder = Encoder(cfg1['stage_1_latent_dim'], cfg1['stage_1_img_size'])
    encoder.load_state_dict(torch.load("saved_models/feature_extractor/hae_encoder.pth")) # Load the saved encoder weights.
    encoder.to(device)
    quantum_layer = get_quantum_torch_layer(cfg1['stage_1_latent_dim'])
    pqc_weights = np.load("saved_models/feature_extractor/hae_pqc_weights.npy") # Load the saved PQC weights.
    quantum_layer.weight = torch.nn.Parameter(torch.Tensor(pqc_weights))
    quantum_layer.to(device)
    
    # --- Generate the dataset for this stage using the loaded feature extractor ---
    print("Generating quantum-enhanced features for downstream task...")
    train_loader, _ = get_data_loaders(batch_size=128, n_samples=cfg2['stage_2_n_samples'], img_size=cfg1['stage_1_img_size'])
    features, labels = generate_quantum_features(encoder, quantum_layer, train_loader, device)
    print(f"Generated {features.shape[0]} feature vectors of dimension {cfg1['stage_1_latent_dim']}.")
    
    # --- Formulate the classical optimization problem (QUBO) ---
    space = cfg2['hyperparameter_space']        # Get the hyperparameter search space from the config.
    objective_function = partial(evaluate_model, quantum_features=features, labels=labels) # Prepare the evaluation function.
    qubo_matrix = create_qubo(objective_function, space) # Create the QUBO matrix by testing all HPO combinations.

    # --- Convert the classical problem into a quantum format (Ising Hamiltonian) ---
    qp = QuadraticProgram()                     # Initialize a Qiskit optimization problem.
    num_qubits = qubo_matrix.shape[0]
    for i in range(num_qubits):
        qp.binary_var(name=f'x{i}')             # Define the binary variables (qubits).
    qp.minimize(quadratic=qubo_matrix)          # Set the QUBO as the problem to be minimized.
    hamiltonian, offset = qp.to_ising()         # Convert the problem into a quantum Hamiltonian.

    # --- Configure and run the QAOA quantum algorithm ---
    print("\nConfiguring and running QAOA for HPO...")
    optimizer = COBYLA(maxiter=50)              # A classical optimizer that fine-tunes the QAOA circuit.
    sampler = SamplerV2()                       # The fast local quantum simulator.
    
    ansatz_template = QAOAAnsatz(hamiltonian, reps=cfg2['stage_2_qaoa_reps']) # Create the abstract QAOA circuit blueprint.
    
    # Use transpile to break down all abstract gates into simple ones the simulator understands.
    decomposed_ansatz = transpile(ansatz_template, basis_gates=['u', 'cx'])
    
    # Assemble the final quantum solver.
    qaoa_solver = SamplingVQE(sampler=sampler, ansatz=decomposed_ansatz, optimizer=optimizer)
    result = qaoa_solver.compute_minimum_eigenvalue(hamiltonian) # Run the quantum optimization.
    
    # --- Decode and display the results from QAOA ---
    best_solution_binary = result.best_measurement['bitstring'] # Get the optimal solution as a binary string.
    best_hyperparams = binary_to_hyperparams(best_solution_binary, space) # Convert the binary string to readable hyperparameters.
    
    print("\n--- QAOA HPO Results ---")
    print(f"Optimal binary string found by QAOA: {best_solution_binary}")
    print(f"Decoded optimal hyperparameters: {best_hyperparams}")
    final_accuracy = evaluate_model(best_hyperparams, features, labels) # Re-run evaluation with the best params to confirm accuracy.
    print(f"Final model accuracy with optimal hyperparameters: {final_accuracy:.4f}")

    # --- Log results to MLflow ---
    mlflow.log_metric("stage_2_final_accuracy", final_accuracy) # Log the final accuracy score.
    mlflow.log_params({f"stage_2_best_{key}": val for key, val in best_hyperparams.items()}) # Log the best hyperparameters found.
    
    # --- Train and save the final, optimized classifier model ---
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
    
    final_model.train()                         # Set the model to training mode.
    for epoch in range(10):                     # Train for 10 full epochs.
        for data, target in train_loader_final:
            optimizer_final.zero_grad()
            output = final_model(data.to(device))
            loss = criterion(output, target.to(device))
            loss.backward()
            optimizer_final.step()
    
    save_dir = "saved_models/tuned_classifier"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "tuned_classifier.pth")
    torch.save(final_model.state_dict(), model_path) # Save the trained model's weights.
    print(f"Final tuned classifier saved to: {model_path}")

    print("Logging tuned classifier artifact to MLflow...")
    mlflow.log_artifact(model_path, artifact_path="stage_2_tuned_classifier") # Save the model file to MLflow.
    print("Logging tuned classifier artifact to MLflow...") # This line is a duplicate but is kept as requested.
    mlflow.log_artifact(model_path, artifact_path="stage_2_tuned_classifier") # It logs the same artifact a second time.

    print("\n--- Hyperparameter Tuning Stage Complete ---")