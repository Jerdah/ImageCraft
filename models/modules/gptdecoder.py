import torch
import torch.nn as nn

from models.modules.gptdecoderblock import GPTDecoderBlock

class GPTDecoder(nn.Module):
    """
    GPT decoder.
    """
    def __init__(self, config) -> None:

        super().__init__()
        self.decoder_blocks = nn.ModuleList([GPTDecoderBlock(config) for _ in range(config["num_decoders"])])
    
    def forward(self, x: torch.Tensor, image_encoding: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:

        for block in self.decoder_blocks:
            x = block(x, image_encoding, attn_mask)
        
        return x
    