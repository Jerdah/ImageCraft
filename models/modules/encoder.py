import torch
import torch.nn as nn

from models.modules.encoderblock import EncoderBlock

class Encoder(nn.Module):
    """
    The Vision Transformer (ViT) encoder.
    """
    def __init__(self, config) -> None:

        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config["num_encoders"])])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for block in self.blocks:
            x = block(x)
        
        return x
