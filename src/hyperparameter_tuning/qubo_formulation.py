import numpy as np

def define_hyperparameter_space():
    """Defines the discrete search space for each hyperparameter."""
    space = {
        'hidden_dim': [32, 64, 128],
        'lr': [0.01, 0.005, 0.001],
        'dropout': [0.2, 0.4, 0.6]
    }
    return space

def binary_to_hyperparams(binary_string, space):
    """Converts a binary string solution from QAOA back to hyperparameter values."""
    if len(binary_string) != 6:
        raise ValueError("Binary string must have length 6.")
        
    hd_bin, lr_bin, dr_bin = binary_string[0:2], binary_string[2:4], binary_string[4:6]
    hd_idx, lr_idx, dr_idx = int(hd_bin, 2), int(lr_bin, 2), int(dr_bin, 2)
    
    hyperparams = {
        'hidden_dim': space['hidden_dim'][hd_idx] if hd_idx < len(space['hidden_dim']) else space['hidden_dim'][-1],
        'lr': space['lr'][lr_idx] if lr_idx < len(space['lr']) else space['lr'][-1],
        'dropout': space['dropout'][dr_idx] if dr_idx < len(space['dropout']) else space['dropout'][-1]
    }
    return hyperparams

def create_qubo(objective_function, space):
    """Creates the QUBO matrix by evaluating the objective function."""
    num_qubits = 6
    num_combinations = 2**num_qubits
    qubo_matrix = np.zeros((num_qubits, num_qubits))
    linear_terms = np.zeros(num_qubits)

    print("Evaluating objective function across search space to build QUBO...")
    for i in range(num_combinations):
        binary_string = format(i, f'0{num_qubits}b')
        try:
            hyperparams = binary_to_hyperparams(binary_string, space)
            score = -objective_function(hyperparams)
            num_set_bits = sum(int(c) for c in binary_string)
            if num_set_bits > 0:
                for j in range(num_qubits):
                    if binary_string[j] == '1':
                        linear_terms[j] += score / num_set_bits
        except IndexError:
            continue
    
    np.fill_diagonal(qubo_matrix, linear_terms)
    print("QUBO matrix construction complete.")
    return qubo_matrix