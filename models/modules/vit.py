import torch
import torch.nn as nn

from models.modules.encoder import Encoder
from models.modules.vitembedding import ViTEmbedding

class ViT(nn.Module):
    """
    Vision Transformer (ViT) model.
    """
    def __init__(self, config) -> None:

        super().__init__()
        
        self.embedding_layer = ViTEmbedding(config)
        self.encoder = Encoder(config)

    def forward(self, images: torch.Tensor) -> torch.Tensor:

        embeddings = self.embedding_layer(images)
        
        encoded_vectors = self.encoder(embeddings)

        return encoded_vectors[:, 0, :]

