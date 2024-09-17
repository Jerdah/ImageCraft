import torch

# Configuration Parameters
img_size = 224  # Image size (height and width) in pixels
ctx_length = 256  # Length of the context (sequence length) for GPT
num_encoders_vit = 8  # Number of encoder layers in the Vision Transformer
num_heads_vit = 4  # Number of attention heads in the Vision Transformer
ps = 16  # Patch size (height and width) for the Vision Transformer
c = 3  # Number of color channels (RGB) in the image
d_model_vit = ps**2 * c  # Dimension of the model (embedding dimension) for the Vision Transformer
num_patches = (img_size * img_size) // (ps * ps)  # Number of patches in the input image
d_model_gpt = 512  # Dimension of the model (embedding dimension) for GPT
num_decoders_gpt = 8  # Number of decoder layers in GPT
num_heads_gpt = 8  # Number of attention heads in GPT
softmax_denom_eps = 1e-9  # Epsilon for numerical stability in softmax calculation
device = "cuda" if torch.cuda.is_available() else "cpu"  # Device to use (GPU if available, otherwise CPU)
attn_dropout = 0.25  # Dropout probability for attention layers
mlp_dropout = 0.25  # Dropout probability for the MLP layers
emb_dropout = 0.25  # Dropout probability for the embedding layers

# Vision Transformer Configuration
vit_kwargs = {
    "num_encoders": num_encoders_vit,  # Number of encoder layers
    "num_heads": num_heads_vit,  # Number of attention heads
    "num_patches": num_patches,  # Number of patches in the image
    "patch_size": ps,  # Size of each patch
    "channels": c,  # Number of input channels
    "d_model": d_model_vit,  # Dimension of the model (embedding dimension)
    "pretrained_model_name": None,  # Pretrained model name (None means no pretrained model)
    "device": device,  # Device to use (GPU or CPU)
    "emb_dropout": emb_dropout,  # Dropout probability for embedding layers
    "mlp_dropout": mlp_dropout,  # Dropout probability for MLP layers
    "attn_dropout": attn_dropout  # Dropout probability for attention layers
}

# GPT Configuration
gpt_kwargs = {
    "d_model": d_model_gpt,  # Dimension of the model (embedding dimension)
    "context_length": ctx_length,  # Length of the context (sequence length)
    "num_decoders": num_decoders_gpt,  # Number of decoder layers
    "softmax_eps": softmax_denom_eps,  # Epsilon for numerical stability in softmax
    "num_heads": num_heads_gpt,  # Number of attention heads
    "device": device,  # Device to use (GPU or CPU)
    "emb_dropout": emb_dropout,  # Dropout probability for embedding layers
    "mlp_dropout": mlp_dropout,  # Dropout probability for MLP layers
    "attn_dropout": attn_dropout  # Dropout probability for attention layers
    # Add ignore_index and vocab_size before using in the model
}

# Complete Configuration
config = {
    "vit_kwargs": vit_kwargs,  # Vision Transformer configuration
    "gpt_kwargs": gpt_kwargs,  # GPT configuration
    "device": device,  # Device to use (GPU or CPU)
    'img_size': img_size  # Image size
}