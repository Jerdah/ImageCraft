import argparse
from datasets import load_dataset
import src.utils.tools as tools


def download_flickr(raw_dir, processed_dir, dataset_size="10%"):

    dataset_name = "flickr"
    cache_dir = raw_dir + dataset_name
    dataset = load_dataset(
        "nlphuji/flickr30k", split=f"test[:{dataset_size}]", cache_dir=cache_dir
    )

    train_dataset = dataset.filter(lambda data: data["split"].startswith("train"))
    test_dataset = dataset.filter(lambda data: data["split"].startswith("test"))

    train_dataset = train_dataset.select_columns(["image", "caption"])
    test_dataset = test_dataset.select_columns(["image", "caption"])

    train_dataset.save_to_disk(f"{processed_dir}/{dataset_name}/train")
    test_dataset.save_to_disk(f"{processed_dir}/{dataset_name}/test")


def download_coco(raw_dir, processed_dir, dataset_size="10%"):

    dataset_name = "coco"

    cache_dir = raw_dir + dataset_name

    train_dataset = load_dataset(
        "patomp/thai-mscoco-2014-captions",
        split=f"train[:{dataset_size}]",
        cache_dir=cache_dir,
    )
    test_dataset = load_dataset(
        "patomp/thai-mscoco-2014-captions",
        split=f"validation[:{dataset_size}]",
        cache_dir=cache_dir,
    )

    train_dataset = train_dataset.select_columns(["image", "sentences_raw"])
    test_dataset = test_dataset.select_columns(["image", "sentences_raw"])

    train_dataset = train_dataset.rename_column("sentences_raw", "caption")
    test_dataset = test_dataset.rename_column("sentences_raw", "caption")

    train_dataset.save_to_disk(f"{processed_dir}/{dataset_name}/train")
    test_dataset.save_to_disk(f"{processed_dir}/{dataset_name}/test")


def download_dataset(dataset_name, dataset_size="10%"):
    config = tools.load_config()
    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    if dataset_name == "flickr":
        download_flickr(raw_dir, processed_dir, dataset_size)
    else:
        download_coco(raw_dir, processed_dir, dataset_size)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download tool")
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--dataset_size", type=str, default="10%")

    args = parser.parse_args()

    download_dataset(args.dataset, args.dataset_size)
