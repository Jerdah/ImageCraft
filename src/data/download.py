import argparse
import pandas as pd
from datasets import load_dataset
import src.utils.tools as tools


from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import joblib


def transforms(examples):
    examples["pixel_values"] = [
        image.convert("RGB").resize((100, 100)) for image in examples["image"]
    ]
    return examples


def take_one_caption(sample):

    sample["caption"] = sample["caption"][0]

    return sample


def download_flickr(raw_dir, interim_dir, processed_dir, dataset_size="10%"):

    dataset_name = "flickr"
    cache_dir = raw_dir + dataset_name
    dataset = load_dataset(
        "nlphuji/flickr30k", split=f"test[:{dataset_size}]", cache_dir=cache_dir
    )

    train_dataset = dataset.filter(lambda data: data["split"].startswith("train"))
    test_set = dataset.filter(lambda data: data["split"].startswith("test"))

    train_dataset.save_to_disk(f"{interim_dir}/{dataset_name}/train")
    test_set.save_to_disk(f"{interim_dir}/{dataset_name}/test")

    train_dataset = train_dataset.select_columns(["image", "caption"])
    test_dataset = test_set.select_columns(["image", "caption"])

    train_dataset = train_dataset.map(take_one_caption)
    test_dataset = test_dataset.map(take_one_caption)

    train_dataset.save_to_disk(f"{processed_dir}/{dataset_name}/train")
    test_set.save_to_disk(f"{processed_dir}/{dataset_name}/test")


def download_coco(raw_dir, interim_dir, processed_dir, dataset_size="10%"):

    dataset_name = "coco"

    cache_dir = raw_dir + dataset_name

    dataset = load_dataset(
        "shunk031/MSCOCO", split=f"test[:{dataset_size}]", cache_dir=cache_dir
    )

    train_dataset = dataset.filter(lambda data: data["split"].startswith("train"))
    test_set = dataset.filter(lambda data: data["split"].startswith("test"))

    train_dataset.save_to_disk(f"{interim_dir}/{dataset_name}/train")
    test_set.save_to_disk(f"{interim_dir}/{dataset_name}/test")

    train_dataset = train_dataset.select_columns(["image", "caption"])
    test_dataset = test_set.select_columns(["image", "caption"])

    train_dataset = train_dataset.map(take_one_caption)
    test_dataset = test_dataset.map(take_one_caption)

    train_dataset.save_to_disk(f"{processed_dir}/{dataset_name}/train")
    test_set.save_to_disk(f"{processed_dir}/{dataset_name}/test")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download tool")
    parser.add_argument("--dataset", type=str, default="flickr")
    parser.add_argument("--dataset_size", type=str, default="10%")

    args = parser.parse_args()

    config = tools.load_config()
    raw_dir = config["data"]["raw_dir"]
    interim_dir = config["data"]["interim_dir"]
    processed_dir = config["data"]["processed_dir"]
    model_dir = config["model_dir"]

    if args.dataset == "flickr":
        download_flickr(
            raw_dir, interim_dir, processed_dir, dataset_size=args.dataset_size
        )
    else:
        download_coco(
            raw_dir, interim_dir, processed_dir, dataset_size=args.dataset_size
        )
