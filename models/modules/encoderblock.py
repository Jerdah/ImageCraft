import torch
import torch.nn as nn

from models.modules.mlpblock import MLPBlock
from models.modules.msablock import MSABlock

class EncoderBlock(nn.Module):
    """
    Encoder block combining both Multihead Self-Attention and Feed-Forward Network blocks.
    """
    def __init__(self, config) -> None:

        super().__init__()
        self.msa_block = MSABlock(config)
        self.mlp_block = MLPBlock(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.mlp_block(self.msa_block(x))
