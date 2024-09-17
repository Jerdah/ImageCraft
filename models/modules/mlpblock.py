import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    """
    Feed-Forward Network block.
    """
    def __init__(self, config) -> None:

        super().__init__()
        d_model = config["d_model"]

        self.dense_net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(p=config['mlp_dropout']),
            nn.Linear(d_model * 4, d_model)
        )

        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.layer_norm(x + self.dense_net(x))
