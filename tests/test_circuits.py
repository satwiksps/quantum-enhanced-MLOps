import unittest                                     # The standard Python library for creating and running tests.
from qiskit.circuit import QuantumCircuit           # Imports the main QuantumCircuit class for type checking.
from qiskit_machine_learning.connectors import TorchConnector # Imports the TorchConnector for type checking.

# --- Import the functions from your project that you want to test ---
from src.feature_engineering.quantum_circuits import create_pqc, get_quantum_torch_layer

class TestQuantumCircuits(unittest.TestCase):
    """
    Fast tests to ensure quantum circuits are constructed correctly.
    These tests are run by the CI/CD pipeline.
    """
    def test_pqc_creation(self):
        """Test if the PQC is created with the right number of qubits and parameters."""
        n_qubits = 4                                  # Define a test parameter.
        qc, inputs, weights = create_pqc(n_qubits)    # Call the function being tested.
        
        # --- Assertions: Check if the function's output is correct ---
        self.assertIsInstance(qc, QuantumCircuit)     # 1. Is the output actually a QuantumCircuit?
        self.assertEqual(qc.num_qubits, n_qubits)     # 2. Does it have the correct number of qubits?
        self.assertEqual(len(inputs), n_qubits)       # 3. Does it have the correct number of input parameters?

    def test_torch_layer_creation(self):
        """Test if the QNN-Torch layer wrapper is created successfully."""
        n_qubits = 4                                  # Define a test parameter.
        torch_layer = get_quantum_torch_layer(n_qubits) # Call the function being tested.
        
        # --- Assertions: Check if the wrapper is correct ---
        self.assertIsInstance(torch_layer, TorchConnector) # 1. Is the output the correct TorchConnector type?
        # 2. Does the created layer have a 'weight' attribute? This is crucial for training.
        self.assertTrue(hasattr(torch_layer, 'weight'))

# --- Standard Python entry point to make the test file runnable ---
if __name__ == '__main__':
    unittest.main()                               # This command discovers and runs all tests in the file.