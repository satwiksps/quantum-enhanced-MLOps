import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    A classical CNN to encode images into a latent vector.
    """
    def __init__(self, latent_dim, img_size):
        super(Encoder, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        conv_output_size = 16 * 4 * 4
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return torch.tanh(x)

class Decoder(nn.Module):
    """
    A classical network to reconstruct images from the PQC's output.
    """
    def __init__(self, latent_dim, img_size):
        super(Decoder, self).__init__()
        self.img_size = img_size
        conv_input_size = 16 * 4 * 4
        self.fc_block = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, conv_input_size),
            nn.ReLU()
        )
        self.deconv_block = nn.Sequential(
            # CORRECTED PARAMETERS: This combination ensures a 4x4 -> 7x7 upsampling.
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # CORRECTED PARAMETERS: This combination ensures a 7x7 -> 14x14 upsampling.
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc_block(x)
        x = x.view(x.size(0), 16, 4, 4)
        x = self.deconv_block(x)
        return x