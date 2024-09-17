import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
from typing import Tuple

from data.captiondataset import CaptionDataset


def process_flickr30k(csv_file_path: str, image_folder: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file_path, delimiter="|")
    # df.drop_duplicates(subset=['image_name'], inplace=True)
    df.drop(columns=" comment_number", axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.rename({" comment": "caption"}, axis=1, inplace=True)
    df.iloc[:, 0] = image_folder + "/" + df.iloc[:, 0]
    df["caption"] = "[BOS] " + df["caption"] + " [EOS]"

    return df


def process_and_save_flickr30k(csv_file_path: str, image_folder: str):
    dataset_path = "temp/preprocessed_data.csv"
    df = pd.read_csv(csv_file_path, delimiter="|")
    # df.drop_duplicates(subset=['image_name'], inplace=True)
    df.drop(columns=" comment_number", axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.rename({" comment": "caption"}, axis=1, inplace=True)
    df.iloc[:, 0] = image_folder + "/" + df.iloc[:, 0]
    df["caption"] = "[BOS] " + df["caption"] + " [EOS]"
    df.to_csv(dataset_path, index=False)
    return dataset_path


def train_test_split(
    train_config: dict, model_config: dict, data: pd.DataFrame, tokenizer
) -> Tuple[DataLoader, DataLoader]:
    idxs = set(range(data.shape[0]))

    # Randomly split indices for training and testing
    train_idxs = random.sample(
        sorted(idxs), k=int(len(idxs) * train_config["train_size"])
    )
    test_idxs = list(idxs.difference(set(train_idxs)))

    # Split data into training and testing sets
    train_data = data.copy(deep=True).iloc[train_idxs, :].reset_index(drop=True)
    test_data = data.copy(deep=True).iloc[test_idxs, :].reset_index(drop=True)

    # Create dataset objects for training and testing
    train_dataset = CaptionDataset(
        dataframe=train_data,
        image_size=model_config["img_size"],
        context_length=model_config["gpt_kwargs"]["context_length"],
        tokenizer=tokenizer,
    )

    test_dataset = CaptionDataset(
        dataframe=test_data,
        image_size=model_config["img_size"],
        context_length=model_config["gpt_kwargs"]["context_length"],
        tokenizer=tokenizer,
    )

    # Create DataLoader objects for training and testing datasets
    train_dl = DataLoader(
        dataset=train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=2,
    )

    test_dl = DataLoader(
        dataset=test_dataset, batch_size=train_config["batch_size"], shuffle=False
    )

    return train_dl, test_dl
