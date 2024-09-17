import torch
import torch.nn as nn
import math

class CausalSelfAttnBlock(nn.Module):
    """
    GPT causal self-attention block.
    """
    def __init__(self, config) -> None:

        super().__init__()
        assert config["d_model"] % config["num_heads"] == 0, \
            ValueError(f"{config['d_model']} d_model should be exactly divisible by {config['num_heads']} num_heads")
        
        self.d_model = config["d_model"]
        self.head_dim = config["d_model"] // config["num_heads"]
        self.num_heads = config["num_heads"]
        self.softmax_eps = config["softmax_eps"]
        
        self.projection_layer = nn.Linear(self.d_model, self.d_model * 3)
        self.out_layer = nn.Linear(self.d_model, self.d_model)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.d_model)
        self.attn_dropout = nn.Dropout(p=config['attn_dropout'])

    def _safe_softmax(self, x: torch.Tensor) -> torch.Tensor:

        num = torch.exp(x)
        denom = torch.exp(x).sum(dim=-1, keepdim=True) + self.softmax_eps
        return num / denom
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:

        B, CTX_LENGTH = x.shape[0], x.shape[1]
        q, k, v = self.projection_layer(x).split(self.d_model, dim=2)  # B, CTX_LENGTH, d_model
        q = q.view(B, CTX_LENGTH, self.num_heads, self.head_dim).transpose(1, 2)  # B, num_heads, CTX_LENGTH, head_dim
        k = k.view(B, CTX_LENGTH, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, CTX_LENGTH, self.num_heads, self.head_dim).transpose(1, 2)
        
        q_k_prod = (q @ k.transpose(2, 3)) + attn_mask.unsqueeze(1)  # B, num_heads, CTX_LENGTH, CTX_LENGTH
        wts = self._safe_softmax(q_k_prod / math.sqrt(self.head_dim))  # B, num_heads, CTX_LENGTH, CTX_LENGTH
        wts = self.attn_dropout(wts)
        attn_outputs = wts @ v  # B, num_heads, CTX_LENGTH, head_dim
        y = attn_outputs.transpose(1, 2).contiguous().view(B, CTX_LENGTH, -1)
        return self.layer_norm(x + self.out_layer(y))

