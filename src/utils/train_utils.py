import torch

PROMPT = "<image>Describe en<bos>"


def train_collate_fn(examples, processor, device):
    images = [example["image"] for example in examples]
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
    captions = inputs["labels"]

    return input_ids, token_type_ids, attention_mask, pixel_values, captions


def eval_collate_fn(examples, processor, device):
    images = [example["image"] for example in examples]
    texts = [PROMPT for _ in examples]
    captions = [example["caption"][0] for example in examples]

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

    return input_ids, attention_mask, pixel_values, captions
