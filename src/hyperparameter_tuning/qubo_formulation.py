import numpy as np                                  # The fundamental library for numerical operations.


def binary_to_hyperparams(binary_string, space):
    """Converts a binary string solution from QAOA back to hyperparameter values."""
    if len(binary_string) != 6:                       # Basic error check for the input string length.
        raise ValueError("Binary string must have length 6.")
        
    # --- Slice the 6-bit string into three 2-bit chunks ---
    hd_bin, lr_bin, dr_bin = binary_string[0:2], binary_string[2:4], binary_string[4:6]
    
    # --- Convert each 2-bit chunk from a binary string to an integer index (0, 1, 2, or 3) ---
    hd_idx, lr_idx, dr_idx = int(hd_bin, 2), int(lr_bin, 2), int(dr_bin, 2)
    
    # --- Use the integer index to look up the actual hyperparameter value from the 'space' dictionary ---
    # This safely handles cases where the index might be out of range.
    hyperparams = {
        'hidden_dim': space['hidden_dim'][hd_idx] if hd_idx < len(space['hidden_dim']) else space['hidden_dim'][-1],
        'lr': space['learning_rate'][lr_idx] if lr_idx < len(space['learning_rate']) else space['learning_rate'][-1],
        'dropout': space['dropout'][dr_idx] if dr_idx < len(space['dropout']) else space['dropout'][-1]
    }
    return hyperparams

# The function now accepts the 'space' dictionary as an argument from the config file.
def create_qubo(objective_function, space):
    """Creates the QUBO matrix by evaluating the objective function across the provided space."""
    num_qubits = 6                                  # We need 2 qubits for each of the 3 hyperparameters.
    num_combinations = 2**num_qubits              # Total possible combinations (2^6 = 64).
    qubo_matrix = np.zeros((num_qubits, num_qubits)) # Initialize the QUBO matrix with zeros.
    linear_terms = np.zeros(num_qubits)             # Initialize the linear terms (diagonal of the QUBO).

    print("Evaluating objective function across search space to build QUBO...")
    # --- Loop through every possible combination of hyperparameters ---
    for i in range(num_combinations):
        binary_string = format(i, f'0{num_qubits}b')  # Convert the loop counter to a 6-bit binary string (e.g., '000000', '000001').
        try:
            hyperparams = binary_to_hyperparams(binary_string, space) # Convert the binary string to a set of hyperparameters.
            
            # Run the model evaluation and get a score. We negate it because QAOA finds a minimum.
            score = -objective_function(hyperparams) 
            
            # --- This logic assigns the score to the appropriate terms in the QUBO formulation ---
            for j in range(num_qubits):
                if binary_string[j] == '1':         # If the j-th qubit is in the '1' state...
                    linear_terms[j] += score        # ...add the score to its corresponding linear term.
            
            # This is where you would add logic for interaction terms (the off-diagonal elements).
            # For this simple mapping, there are no interactions.
            for j in range(num_qubits):
                for k in range(j + 1, num_qubits):
                    if binary_string[j] == '1' and binary_string[k] == '1':
                        pass # No quadratic interaction terms in this simple formulation.
        
        except IndexError:
            # This handles cases where the binary index is invalid for a non-power-of-2 search space.
            continue
    
    # Place the calculated linear terms on the diagonal of the QUBO matrix.
    np.fill_diagonal(qubo_matrix, linear_terms)
    print("QUBO matrix construction complete.")
    return qubo_matrix                              # Return the final QUBO matrix.