import torch
import torch.nn as nn

from models.modules.gptdecoder import GPTDecoder
from models.modules.gptembedding import GPTEmbedding

class GPT(nn.Module):
    """
    GPT model for image caption generation.
    """
    def __init__(self, config) -> None:

        super().__init__()
        self.device = config["device"]
        self.context_length = config["context_length"]
        self.softmax_eps = config["softmax_eps"]
        self.embedding = GPTEmbedding(config)
        self.decoder = GPTDecoder(config)
        self.cls_head = nn.Linear(config["d_model"], config["vocab_size"])
        self.cls_head.weight = self.embedding.token_embedding.weight
        # Removed weight tying as it led to slower convergence
        self.ignore_index = config["ignore_index"]
    
    def _create_mask(self, context_length: int, attn_mask: torch.Tensor) -> torch.Tensor:

        mask = torch.triu(
            input=torch.ones(size=(context_length, context_length), requires_grad=False) * float("-inf"),
            diagonal=1
        ).unsqueeze(0).repeat(attn_mask.shape[0], 1, 1)
        mask = mask.to(self.device)
        for i in range(mask.shape[0]):
            mask[i, attn_mask[i].logical_not(), :] = float("-inf")
        return mask  # B, CTX_LENGTH, CTX_LENGTH
        
    def forward(self, tokens: torch.Tensor, image_encoding: torch.Tensor, attn_mask: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor]:

        embeddings = self.embedding(tokens)  # B, CTX_LENGTH, d_model
        mask = self._create_mask(tokens.shape[1], attn_mask)
        decoder_out = self.decoder(embeddings, image_encoding, mask)  # B, CTX_LENGTH, d_model
        logits = self.cls_head(decoder_out)  # B, CTX_LENGTH, vocab_size
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=self.ignore_index)
        
        return logits, loss

