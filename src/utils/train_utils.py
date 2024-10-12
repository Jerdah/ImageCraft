from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from huggingface_hub import HfApi

from src.model.modules.gemma import KVCache
from src.utils import tools

PROMPT = "<image><bos>caption en"


def train_collate_fn(examples, processor, device):
    images = [example["image"].convert("RGB") for example in examples]
    texts = [PROMPT for _ in examples]
    captions = [example["caption"][0] for example in examples]

    inputs = processor(
        text=texts,
        images=images,
        suffix=captions,
        return_tensors="pt",
        padding="longest",
        do_convert_rgb=True,
    ).to(device)

    inputs = inputs.to(torch.bfloat16).to(device)

    input_ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    labels = inputs["labels"]

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels,
    }


def eval_collate_fn(examples, processor, device):
    images = [example["image"].convert("RGB") for example in examples]
    texts = [PROMPT for _ in examples]
    captions = [example["caption"] for example in examples]

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding="longest",
        do_convert_rgb=True,
    ).to(device)

    inputs = inputs.to(torch.bfloat16).to(device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": captions,
    }


def save_to_hub(model, tokenizer, repository, commit_message):
    api = HfApi()
    with TemporaryDirectory() as tmp_dir:
        model = model.merge_and_unload()
        model.save_pretrained(tmp_dir, safe_serialization=False)
        tokenizer.save_pretrained(tmp_dir, safe_serialization=False)

        # Push to Hub
        api.upload_folder(
            folder_path=tmp_dir,
            repo_id=repository,
            commit_message=commit_message,
        )
