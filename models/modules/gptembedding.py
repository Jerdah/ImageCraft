import torch
import torch.nn as nn

class GPTEmbedding(nn.Module):
    """
    GPT decoder embedding class.
    """
    def __init__(self, config) -> None:

        super().__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings=config["vocab_size"],
            embedding_dim=config["d_model"]
        )
        
        self.positional_encoding = nn.Parameter(
            data=torch.randn(size=(1, config["context_length"], config["d_model"])),
            requires_grad=True
        )
        self.dropout = nn.Dropout(p=config['emb_dropout'])
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:

        token_embeddings = self.token_embedding(tokens)
        return self.dropout(self.positional_encoding[:, :tokens.shape[1], :] + token_embeddings)

