import torch
import torch.nn as nn

from models.modules.causalselfattnblock import CausalSelfAttnBlock
from models.modules.crossattnblock import CrossAttnBlock
from models.modules.mlpblock import MLPBlock

class GPTDecoderBlock(nn.Module):
    """
    GPT decoder block.
    """
    def __init__(self, config) -> None:

        super().__init__()
        self.csa_block = CausalSelfAttnBlock(config)
        self.cross_attn_block = CrossAttnBlock(config)
        self.mlp_block = MLPBlock(config)
    
    def forward(self, x: torch.Tensor, image_encoding: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:

        csa_out = self.csa_block(x, attn_mask)
        cross_out = self.cross_attn_block(csa_out, image_encoding)
        mlp_out = self.mlp_block(cross_out)
        return mlp_out
    