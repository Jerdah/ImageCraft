import torch
import torch.nn as nn

class PatchEmbeddings(nn.Module):
    """
    Extract patch embeddings from input images using a convolutional layer.
    """
    def __init__(self, config):

        super().__init__()

        # Convolutional layer to create patch embeddings
        self.conv_patch_layer = nn.Conv2d(
            in_channels=config['channels'],
            out_channels=config['d_model'],
            kernel_size=config['patch_size'],
            stride=config['patch_size']
        )

        # Flatten patches into a 2D tensor for further processing
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Apply convolution to extract patch embeddings
        patched_tensor = self.conv_patch_layer(x)
        
        # Flatten the patched tensor
        flattend_tensor = self.flatten(patched_tensor)
        
        # Permute dimensions to match (B, num_patches, d_model) format
        return flattend_tensor.permute(0, 2, 1)
