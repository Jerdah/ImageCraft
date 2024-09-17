import torch
import torch.nn as nn

class MSABlock(nn.Module):
    """
    Multihead Self-Attention block.
    """
    def __init__(self, config) -> None:

        super().__init__()

        # Multihead self-attention layer
        self.attn_block = nn.MultiheadAttention(
            embed_dim=config["d_model"],
            num_heads=config["num_heads"],
            batch_first=True,
            dropout=config['attn_dropout']
        )
        
        # Layer normalization for attention output
        self.layer_norm = nn.LayerNorm(normalized_shape=config["d_model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        attn_output, _ = self.attn_block(x, x, x)
        
        return self.layer_norm(x + attn_output)
