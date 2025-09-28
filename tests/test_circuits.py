import unittest
from qiskit.circuit import QuantumCircuit
from qiskit_machine_learning.connectors import TorchConnector
from src.feature_engineering.quantum_circuits import create_pqc, get_quantum_torch_layer

class TestQuantumCircuits(unittest.TestCase):
    """
    Fast tests to ensure quantum circuits are constructed correctly.
    """
    def test_pqc_creation(self):
        """Test if the PQC is created with the right number of qubits and parameters."""
        n_qubits = 4
        qc, inputs, weights = create_pqc(n_qubits)
        self.assertIsInstance(qc, QuantumCircuit)
        self.assertEqual(qc.num_qubits, n_qubits)
        self.assertEqual(len(inputs), n_qubits)

    def test_torch_layer_creation(self):
        """Test if the QNN-Torch layer wrapper is created successfully."""
        n_qubits = 4
        torch_layer = get_quantum_torch_layer(n_qubits)
        self.assertIsInstance(torch_layer, TorchConnector)
        # Check if it has weights that can be trained
        self.assertTrue(hasattr(torch_layer, 'weight'))

if __name__ == '__main__':
    unittest.main()