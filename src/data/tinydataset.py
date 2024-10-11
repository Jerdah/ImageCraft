import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

import torchvision.transforms as T

from src.utils.model_utils import get_model_inputs

PROMPT = "caption en"


class TinyDataset(Dataset):
    def __init__(self, data_dir, processor):
        self.processor = processor
        self.data_dir = data_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(os.path.join(data_dir, "data.json"), "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]
        image_path = os.path.join(self.data_dir, item["image_file"])
        suffix = item["text"]
        image = Image.open(image_path)

        inputs = self.processor(
            text=PROMPT,
            images=image,
            suffix=suffix,
            return_tensors="pt",
            max_length=100,
            padding="max_length",
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        labels = inputs["labels"]

        print(f"Pixel values shape: {pixel_values.shape}")
        print(f"Labels: {labels.shape}")

        # inputs = get_model_inputs(self.processor, PROMPT, text, image, self.device)

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }
