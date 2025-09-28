import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

def create_pqc(n_qubits, n_layers=2):
    """
    Creates a Parameterized Quantum Circuit (PQC) to act as a quantum bottleneck.
    """
    input_params = ParameterVector('input', n_qubits)
    weight_params = ParameterVector('weights', n_qubits * n_layers * 2)
    qc = QuantumCircuit(n_qubits)
    
    for i in range(n_qubits):
        qc.ry(np.pi * input_params[i], i)

    weight_idx = 0
    for _ in range(n_layers):
        for i in range(n_qubits):
            qc.ry(weight_params[weight_idx], i)
            weight_idx += 1
        for i in range(n_qubits):
            qc.rz(weight_params[weight_idx], i)
            weight_idx += 1
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.cx(n_qubits - 1, 0)
        qc.barrier()
        
    return qc, input_params, weight_params

def create_qnn(n_qubits):
    """
    Creates a Quantum Neural Network (QNN) using the PQC.
    """
    qc, inputs, weights = create_pqc(n_qubits)
    
    # CORRECTED LOGIC: This new logic creates a list of Pauli strings where each is n_qubits long.
    # e.g., for n_qubits=4, it produces ['ZIII', 'IZII', 'IIZI', 'IIIZ']
    pauli_strings = ['I'*i + 'Z' + 'I'*(n_qubits-i-1) for i in range(n_qubits)]
    observables = [SparsePauliOp(s) for s in pauli_strings]

    qnn = EstimatorQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        observables=observables
    )
    return qnn

def get_quantum_torch_layer(n_qubits):
    """
    Wraps the QNN in a TorchConnector to make it a PyTorch layer.
    """
    qnn = create_qnn(n_qubits)
    initial_weights = np.zeros(qnn.num_weights)
    return TorchConnector(qnn, initial_weights=initial_weights)