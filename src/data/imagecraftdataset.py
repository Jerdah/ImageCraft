import torch
from torch.utils.data import Dataset

from src.utils.model_utils import get_model_inputs


PROMPT = "<image><bos>caption en"


class ImageCraftDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        caption = item["caption"][0]

        image = item["image"]

        inputs = self.processor(
            text=[PROMPT],
            images=[image],
            suffix=caption,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            do_convert_rgb=True,
        ).to(self.device)

        image.close()

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
