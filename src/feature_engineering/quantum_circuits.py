import numpy as np                                      # For numerical operations, especially for pi.
from qiskit import QuantumCircuit                       # The main class for building quantum circuits.
from qiskit.circuit import ParameterVector              # Used to create symbolic parameters in circuits.
from qiskit.quantum_info import SparsePauliOp           # A format for defining quantum observables (measurements).
from qiskit_machine_learning.connectors import TorchConnector # The bridge between Qiskit and PyTorch.
from qiskit_machine_learning.neural_networks import EstimatorQNN # A Qiskit object that represents a Quantum Neural Network.

def create_pqc(n_qubits, n_layers=2):
    """
    Creates a Parameterized Quantum Circuit (PQC) to act as a quantum bottleneck.
    """
    input_params = ParameterVector('input', n_qubits)     # Placeholders for the input data from the encoder.
    weight_params = ParameterVector('weights', n_qubits * n_layers * 2) # Placeholders for the trainable weights.
    qc = QuantumCircuit(n_qubits)                       # Initialize the quantum circuit with `n_qubits`.
    
    # --- 1. Encoding Layer ---
    # Encode the classical input data into the quantum state by rotating qubits.
    for i in range(n_qubits):
        qc.ry(np.pi * input_params[i], i)               # Apply a Y-rotation gate to each qubit.

    # --- 2. Trainable Ansatz / Variational Layers ---
    weight_idx = 0                                      # Initialize a counter for the weights.
    for _ in range(n_layers):                           # Repeat the following block for each layer.
        # -- Rotation Layer --
        for i in range(n_qubits):
            qc.ry(weight_params[weight_idx], i)         # Apply a trainable Y-rotation.
            weight_idx += 1
        for i in range(n_qubits):
            qc.rz(weight_params[weight_idx], i)         # Apply a trainable Z-rotation.
            weight_idx += 1
        # -- Entanglement Layer --
        for i in range(n_qubits - 1):                   # Apply CNOT gates to entangle adjacent qubits.
            qc.cx(i, i + 1)
        qc.cx(n_qubits - 1, 0)                          # Entangle the last and first qubits (circular entanglement).
        qc.barrier()                                    # A visual separator in circuit diagrams.
        
    return qc, input_params, weight_params              # Return the circuit and its parameter definitions.

def create_qnn(n_qubits):
    """
    Creates a Quantum Neural Network (QNN) using the PQC.
    """
    qc, inputs, weights = create_pqc(n_qubits)          # Get the parameterized circuit blueprint.
    
    # --- Define How to Measure the Circuit ---
    # This defines 'n_qubits' different measurements, one for each qubit.
    # We will measure the Z-Pauli expectation value for each qubit.
    pauli_strings = ['I'*i + 'Z' + 'I'*(n_qubits-i-1) for i in range(n_qubits)]
    observables = [SparsePauliOp(s) for s in pauli_strings] # Convert the strings to formal Qiskit objects.

    # --- Assemble the Quantum Neural Network ---
    qnn = EstimatorQNN(
        circuit=qc,                                     # The quantum circuit to use.
        input_params=inputs,                            # The parameters to be set by the input data.
        weight_params=weights,                          # The parameters to be trained.
        observables=observables                         # How to measure the circuit to get the output.
    )
    return qnn

def get_quantum_torch_layer(n_qubits):
    """
    Wraps the QNN in a TorchConnector to make it a PyTorch layer.
    """
    qnn = create_qnn(n_qubits)                        # Create the Qiskit QNN object.
    initial_weights = np.zeros(qnn.num_weights)       # Create a starting set of weights (all zeros).
    
    # The TorchConnector is the magic bridge that makes the QNN look and act
    # like a standard PyTorch nn.Module, allowing for backpropagation.
    return TorchConnector(qnn, initial_weights=initial_weights)