import unittest                                 # The standard Python library for creating and running tests.
import torch                                    # The main PyTorch library, needed for creating dummy tensors.

# --- Import the classes from your project that you want to test ---
from src.feature_engineering.classical_components import Encoder, Decoder

class TestClassicalComponents(unittest.TestCase):
    """
    Fast tests to ensure classical models can be initialized and can process data.
    These tests are run by the CI/CD pipeline to catch basic errors quickly.
    """
    def test_encoder(self):
        """Test if the Encoder can process a dummy tensor without crashing."""
        # --- Arrange: Set up the test parameters and objects ---
        latent_dim = 4
        img_size = 14
        encoder = Encoder(latent_dim, img_size)   # Initialize an instance of the Encoder.
        
        # Create a dummy batch of 2 random images with the expected input shape.
        dummy_input = torch.randn(2, 1, img_size, img_size)
        
        # --- Act: Perform the action to be tested ---
        output = encoder(dummy_input)             # Pass the dummy data through the encoder.
        
        # --- Assert: Check if the result is correct ---
        # Verify that the output tensor has the correct shape: (batch_size, latent_dim).
        self.assertEqual(output.shape, (2, latent_dim))

    def test_decoder(self):
        """Test if the Decoder can process a dummy tensor without crashing."""
        # --- Arrange: Set up the test parameters and objects ---
        latent_dim = 4
        img_size = 14
        decoder = Decoder(latent_dim, img_size)   # Initialize an instance of the Decoder.
        
        # Create a dummy batch of 2 random latent vectors with the expected input shape.
        dummy_input = torch.randn(2, latent_dim)
        
        # --- Act: Perform the action to be tested ---
        output = decoder(dummy_input)             # Pass the dummy data through the decoder.
        
        # --- Assert: Check if the result is correct ---
        # Verify that the output tensor (the reconstructed image) has the correct shape.
        self.assertEqual(output.shape, (2, 1, img_size, img_size))

# --- Standard Python entry point to make the test file runnable ---
if __name__ == '__main__':
    unittest.main()                           # This command discovers and runs all tests in the file.