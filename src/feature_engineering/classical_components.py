import torch                    # The main PyTorch library for tensors and neural networks.
import torch.nn as nn           # Provides neural network layers (Linear, Conv2d, etc.).

class Encoder(nn.Module):
    """
    A classical CNN to encode images into a latent vector.
    """
    def __init__(self, latent_dim, img_size):
        super(Encoder, self).__init__()
        # --- Convolutional Block: Extracts features and reduces image size ---
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1), # (Input: 14x14) -> (Output: 7x7)
            nn.ReLU(),                                          # Activation function.
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),# (Input: 7x7) -> (Output: 4x4)
            nn.BatchNorm2d(16),                                 # Stabilizes learning.
            nn.ReLU(),                                          # Activation function.
        )
        # --- Fully Connected Block: Compresses features into the latent vector ---
        conv_output_size = 16 * 4 * 4                           # Calculate the size of the flattened features.
        self.fc_block = nn.Sequential(
            nn.Flatten(),                                       # Convert the 2D feature map to a 1D vector.
            nn.Linear(conv_output_size, 32),                    # First linear layer.
            nn.ReLU(),                                          # Activation function.
            nn.Linear(32, latent_dim)                           # Final layer, outputting the latent vector.
        )

    def forward(self, x):
        x = self.conv_block(x)                                  # Pass image through the convolutional layers.
        x = self.fc_block(x)                                    # Pass features through the fully connected layers.
        return torch.tanh(x)                                    # Normalize latent vector values to be between -1 and 1.

class Decoder(nn.Module):
    """
    A classical network to reconstruct images from the PQC's output.
    """
    def __init__(self, latent_dim, img_size):
        super(Decoder, self).__init__()
        self.img_size = img_size
        conv_input_size = 16 * 4 * 4                            # Size must match the encoder's output.
        # --- Fully Connected Block: Expands the latent vector ---
        self.fc_block = nn.Sequential(
            nn.Linear(latent_dim, 32),                          # First layer, takes latent vector as input.
            nn.ReLU(),                                          # Activation function.
            nn.Linear(32, conv_input_size),                     # Expands vector to prepare for deconvolution.
            nn.ReLU()                                           # Activation function.
        )
        # --- Deconvolution Block: Upsamples features back into an image ---
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=0), # (Input: 4x4) -> (Output: 7x7)
            nn.BatchNorm2d(8),                                  # Stabilizes learning.
            nn.ReLU(),                                          # Activation function.
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (Input: 7x7) -> (Output: 14x14)
            nn.Tanh()                                           # Normalize output pixel values to be between -1 and 1.
        )

    def forward(self, x):
        x = self.fc_block(x)                                    # Pass latent vector through fully connected layers.
        x = x.view(x.size(0), 16, 4, 4)                         # Reshape the 1D vector back into a 2D feature map.
        x = self.deconv_block(x)                                # Pass feature map through deconvolution layers to reconstruct the image.
        return x