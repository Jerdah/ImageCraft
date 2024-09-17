import torch
import torch.nn as nn
import timm
from typing import List, Tuple

from models.modules import GPT
from models.modules.vit import ViT

class ImageCaptioner(nn.Module):
    """
    Vision language main class for image captioning.
    """
    
    def __init__(self, config) -> None:

        super().__init__()
        
        self.device = config['device']
        self.is_vit_pretrained = False
        
        # Initialize Vision Transformer
        if config['vit_kwargs']["pretrained_model_name"] is not None:
            self.is_vit_pretrained = True
            self.vit = timm.create_model(
                model_name=config['vit_kwargs']["pretrained_model_name"],
                pretrained=True,
                num_classes=0,
                global_pool='avg'
            )
            config["vit_kwargs"]["d_model"] = self.vit.embed_dim
        else:   
            self.vit = ViT(config['vit_kwargs'])
        
        # Initialize GPT
        self.gpt = GPT(config['gpt_kwargs'])
        
        # Linear layer to map image encoding dimension to GPT's input dimension
        self.dimension_mapping_layer = nn.Linear(config["vit_kwargs"]['d_model'], config["gpt_kwargs"]['d_model'])
        
    def forward(self, image: torch.Tensor, tokens: torch.Tensor, attn_mask: torch.Tensor, targets: torch.Tensor=None) -> Tuple[torch.Tensor]:

        # Encode image
        image_encoding = self.vit(image)  # (B, d_model)
        
        # Map image encoding to GPT's input dimension
        dimension_mapped_image_encoding = self.dimension_mapping_layer(image_encoding[:, None, :])  # (B, 1, d_model)
        
        # Forward pass through GPT
        return self.gpt(tokens, dimension_mapped_image_encoding, attn_mask, targets)

    @torch.inference_mode()
    def generate(self, 
                 image: torch.Tensor, 
                 sos_token: int,
                 eos_token: int,
                 max_len: int=40) -> List[int]:

        # Encode image
        image_encoding = self.vit(image)  # (B, d_model)
        
        # Map image encoding to GPT's input dimension
        dimension_mapped_image_encoding = self.dimension_mapping_layer(image_encoding[:, None, :])  # (B, 1, d_model)
        
        # Initialize tokens with the start-of-sequence token
        tokens = torch.tensor([[sos_token]], requires_grad=False).to(self.device)
        attn_mask = torch.tensor([[1]], requires_grad=False).to(self.device)
        
        while tokens.shape[1] < max_len and tokens[0, -1] != eos_token:
            # Forward pass through GPT
            logits, _ = self.gpt(tokens, dimension_mapped_image_encoding, attn_mask, None)  # (1, N+1, vocab_size)
            
            # Predict the next token
            next_token = torch.argmax(logits[0, -1, :], dim=0).item()
            
            # Append the predicted token to the sequence
            tokens = torch.cat(
                (tokens, torch.tensor([[next_token]], requires_grad=False)),
                dim=-1
            ).to(self.device)
            
            # Update attention mask
            attn_mask = torch.cat(
                (attn_mask, torch.tensor([[1]], requires_grad=False)),
                dim=-1
            ).to(self.device)
        
        return list(tokens[0])
    
    @classmethod
    def from_pretrained(cls, checkpoint, device):

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"{checkpoint} does not exist")

        cp = torch.load(checkpoint, map_location=device)
        
        # Update device information in the model configuration
        cp['model_config']['device'] = device
        cp['model_config']['vit_kwargs']['device'] = device
        cp['model_config']['gpt_kwargs']['device'] = device

        # Initialize model with configuration and load state_dict
        model = cls(cp['model_config'])
        model.load_state_dict(cp['model_state_dict'])
        model = model.to(device)
        
        return model

