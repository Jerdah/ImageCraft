import os

import requests
import torch


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USER"] = "nsandiman"
import argparse

from PIL import Image

from datasets import load_from_disk

from src.model.imagecraft import ImageCraftModel
from src.model.modules.configs import ImageCraftConfig
from src.utils import tools
import torchvision.transforms as transforms


def run():
    # image = data["image"]
    prompt = "Caption the image."

    config = ImageCraftConfig()
    model = ImageCraftModel(config)
    model.from_pretrained("/content/ImageCraft/models/paligemma-3b-pt-224")

    image_path = "media/images/1.jpeg"

    description = model.generate(
        image_path, prompt=prompt, max_tokens=512, do_sample=True
    )
    print(description)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Running inference on imagecraft model."
    )
    parser.add_argument("--dataset", type=str, default="flickr")

    args = parser.parse_args()

    config = tools.load_config()

    config = tools.load_config()
    processed_data_dir = config["data"]["processed_dir"]
    # test_data_path = f"{processed_data_dir}/{args.dataset}/test"
    # test_data = load_from_disk(test_data_path)
    # data = test_data[0]
    run()
