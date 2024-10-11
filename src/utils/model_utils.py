import json
import os
from typing import Optional
from PIL import Image


from src.model.modules.imagecraftconfig import ImageCraftConfig
from src.model.modules.imagecraftprocessor import (
    ImageCraftProcessor,
)


def move_inputs_to_device(model_inputs: dict, device: str):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def get_model_inputs(
    processor: ImageCraftProcessor,
    prompt: str,
    image: Image,
    suffix: Optional[str] = None,
    device: str = "cuda",
):
    images = [image]
    prompts = [prompt]
    if suffix is not None:
        suffix = [suffix]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


def get_config(config_file="config.json"):
    config = None
    with open(config_file, "r") as f:
        model_config_file = json.load(f)
        config = ImageCraftConfig(**model_config_file)

    return config


# def load_hf_model(model_path: str, device: str) -> Tuple[ImageCraft, AutoTokenizer]:

#     # Load the tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
#     assert tokenizer.padding_side == "right"

#     # Find all the *.safetensors files
#     safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

#     # ... and load them one by one in the tensors dictionary
#     tensors = {}
#     for safetensors_file in safetensors_files:
#         with safe_open(safetensors_file, framework="pt", device="cpu") as f:
#             for key in f.keys():
#                 tensors[key] = f.get_tensor(key)

#     # Load the model's config
#     with open(os.path.join(model_path, "config.json"), "r") as f:
#         model_config_file = json.load(f)
#         config = ImageCraftConfig(**model_config_file)

#     # Create the model using the configuration
#     model = ImageCraft(config).to(device)

#     # Load the state dict of the model
#     model.load_state_dict(tensors, strict=False)

#     # Tie weights
#     model.tie_weights()

#     return (model, tokenizer)
