import torch
import torch.nn as nn

from models.modules.patchembeddings import PatchEmbeddings

class ViTEmbedding(nn.Module):
    """
    Create embeddings including positional and class tokens.
    """
    def __init__(self, config):

        super().__init__()

        self.patch_embeddings = PatchEmbeddings(config)

        self.class_token_embedding = nn.Parameter(
            data=torch.randn(size=(1, 1, config['d_model'])),
            requires_grad=True
        )

        self.positional_embedding = nn.Parameter(
            data=torch.randn(size=(1, config['num_patches'] + 1, config['d_model'])),
            requires_grad=True
        )
        
        self.dropout = nn.Dropout(config['emb_dropout'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        patch_embed = self.patch_embeddings(x)

        patch_embeddings_with_class_token = torch.cat(
            tensors=(self.class_token_embedding.repeat(patch_embed.shape[0], 1, 1), patch_embed),
            dim=1
        )

        return self.dropout(patch_embeddings_with_class_token + self.positional_embedding)

