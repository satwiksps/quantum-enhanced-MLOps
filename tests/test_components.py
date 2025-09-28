import unittest
import torch
from src.feature_engineering.classical_components import Encoder, Decoder

class TestClassicalComponents(unittest.TestCase):
    """
    Fast tests to ensure classical models can be initialized and can process data.
    """
    def test_encoder(self):
        """Test if the Encoder can process a dummy tensor without crashing."""
        latent_dim = 4
        img_size = 14
        encoder = Encoder(latent_dim, img_size)
        # Create a dummy batch of 2 images (2, 1, 14, 14)
        dummy_input = torch.randn(2, 1, img_size, img_size)
        output = encoder(dummy_input)
        # Check if the output has the correct shape (batch_size, latent_dim)
        self.assertEqual(output.shape, (2, latent_dim))

    def test_decoder(self):
        """Test if the Decoder can process a dummy tensor without crashing."""
        latent_dim = 4
        img_size = 14
        decoder = Decoder(latent_dim, img_size)
        dummy_input = torch.randn(2, latent_dim)
        output = decoder(dummy_input)
        # Check if the output has the correct shape (batch_size, channels, height, width)
        self.assertEqual(output.shape, (2, 1, img_size, img_size))

if __name__ == '__main__':
    unittest.main()