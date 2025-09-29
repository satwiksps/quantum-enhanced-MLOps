import numpy as np

# The define_hyperparameter_space function is now removed from this file.
# The search space will be read from config.yaml in the main script.

def binary_to_hyperparams(binary_string, space):
    """Converts a binary string solution from QAOA back to hyperparameter values."""
    if len(binary_string) != 6:
        raise ValueError("Binary string must have length 6.")
        
    hd_bin, lr_bin, dr_bin = binary_string[0:2], binary_string[2:4], binary_string[4:6]
    
    # Use integer conversion that is robust to binary strings like '00', '01'
    hd_idx, lr_idx, dr_idx = int(hd_bin, 2), int(lr_bin, 2), int(dr_bin, 2)
    
    # Gracefully handle cases where the binary index might be out of bounds
    hyperparams = {
        'hidden_dim': space['hidden_dim'][hd_idx] if hd_idx < len(space['hidden_dim']) else space['hidden_dim'][-1],
        'lr': space['learning_rate'][lr_idx] if lr_idx < len(space['learning_rate']) else space['learning_rate'][-1],
        'dropout': space['dropout'][dr_idx] if dr_idx < len(space['dropout']) else space['dropout'][-1]
    }
    return hyperparams

# The function now accepts the 'space' dictionary as an argument
def create_qubo(objective_function, space):
    """Creates the QUBO matrix by evaluating the objective function across the provided space."""
    num_qubits = 6 # 2 bits for each of the 3 hyperparameters
    num_combinations = 2**num_qubits
    qubo_matrix = np.zeros((num_qubits, num_qubits))
    linear_terms = np.zeros(num_qubits)

    print("Evaluating objective function across search space to build QUBO...")
    for i in range(num_combinations):
        binary_string = format(i, f'0{num_qubits}b')
        try:
            hyperparams = binary_to_hyperparams(binary_string, space)
            # We want to minimize the *negative* accuracy (i.e., maximize accuracy)
            score = -objective_function(hyperparams) 
            
            # This logic to build the QUBO is simplified and more direct
            for j in range(num_qubits):
                if binary_string[j] == '1':
                    linear_terms[j] += score
            for j in range(num_qubits):
                for k in range(j + 1, num_qubits):
                    if binary_string[j] == '1' and binary_string[k] == '1':
                        # This part is often zero in simple HPO mappings but is included for completeness
                        pass # No quadratic interaction terms in this simple formulation
        except IndexError:
            # This can happen if the number of options in the space is not a power of 2
            continue
    
    np.fill_diagonal(qubo_matrix, linear_terms)
    print("QUBO matrix construction complete.")
    return qubo_matrix