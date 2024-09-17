import torch
import torch.nn as nn

class CrossAttnBlock(nn.Module):
    """
    GPT cross-attention block.
    """
    def __init__(self, config) -> None:

        super().__init__()
        assert config["d_model"] % config["num_heads"] == 0, \
            ValueError(f"{config['d_model']} d_model must be divisible by {config['num_heads']} num_heads")

        self.d_model = config['d_model']
        self.num_heads = config['num_heads']
        self.head_dim = self.d_model // self.num_heads
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.projection_layer = nn.Linear(self.d_model, self.d_model)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.d_model)
        self.attn_dropout = nn.Dropout(p=config['attn_dropout'])
    
    def forward(self, x: torch.Tensor, image_encoding: torch.Tensor) -> torch.Tensor:

        B, CTX_LENGTH, _ = x.shape        

        q = self.q_proj(x).view(B, CTX_LENGTH, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, num_heads, CTX_LENGTH, head_dim
        k = self.k_proj(image_encoding).view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, num_heads, 1, head_dim
        v = self.v_proj(image_encoding).view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, num_heads, 1, head_dim

        wts = F.softmax((q @ k.transpose(2, 3)) / math.sqrt(self.head_dim), dim=-1)  # B, num_heads, CTX_LENGTH, 1
        wts = self.attn_dropout(wts)
        y = wts @ v  # B, num_heads, CTX_LENGTH, head_dim
        y = y.transpose(1, 2).contiguous().view(B, CTX_LENGTH, -1)  # B, CTX_LENGTH, d_model
        return self.layer_norm(x + self.projection_layer(y))
