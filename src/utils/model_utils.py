import glob
import json
import os
from typing import Tuple
from PIL import Image
from safetensors import safe_open


import transformers

from src.model.modules.paligemmaprocessor import PaliGemmaProcessor
from src.model.modules.gemma import (
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
)


def move_inputs_to_device(model_inputs: dict, device: str):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def get_model_inputs(
    processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str
):
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


def get_model_inputs(
    processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str
):
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


def get_attr(obj, names):
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names, val):
    if len(names) == 1:
        return setattr(obj, names[0], val)
    else:
        return set_attr(getattr(obj, names[0]), names[1:], val)


def load_hf_model(
    model_path: str, device: str
) -> Tuple[PaliGemmaForConditionalGeneration, transformers.AutoTokenizer]:

    # Load the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, padding_side="right"
    )
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)
