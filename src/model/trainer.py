import argparse
from functools import partial
import os
from lightning import LightningModule
import numpy as np
import torch

from datasets import load_from_disk


from transformers import PaliGemmaForConditionalGeneration

from transformers import AutoProcessor

from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig


from src.data.download import download_coco, download_flickr
from src.model.modules.configs import ImageCraftConfig
from src.utils import tools

from torch.utils.data import DataLoader

from src.utils.train_utils import eval_collate_fn, train_collate_fn

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import Callback, EarlyStopping

import evaluate


from huggingface_hub import HfApi

api = HfApi()

model_flavor = "nsandiman/imagecraft-ft-fk-224"


class PushToHubCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.model.push_to_hub(
            model_flavor,
            commit_message=f"Training in progress, epoch {trainer.current_epoch}",
        )

    def on_train_end(self, trainer, pl_module):
        pl_module.processor.push_to_hub(model_flavor, commit_message=f"Training done")
        pl_module.model.push_to_hub(model_flavor, commit_message=f"Training done")


early_stop_callback = EarlyStopping(
    monitor="train_loss", patience=3, verbose=False, mode="min"
)


class ImageCraftTrainer(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

    def prepare_data(self):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        env_config = tools.load_config()

        raw_dir = env_config["data"]["raw_dir"]
        interim_dir = env_config["data"]["interim_dir"]
        processed_dir = env_config["data"]["processed_dir"]
        self.train_data_path = f"{processed_dir}/{dataset}/train"
        self.test_data_path = f"{processed_dir}/{dataset}/test"

        if self.config.train_dataset == "flickr":
            download_flickr(
                raw_dir,
                interim_dir,
                processed_dir,
                dataset_size=self.config.train_dataset_size,
            )
        else:
            download_coco(
                raw_dir,
                interim_dir,
                processed_dir,
                dataset_size=self.config.train_dataset_size,
            )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type=torch.bfloat16,
        )
        lora_config = LoraConfig(
            r=8,
            target_modules=[
                "q_proj",
                "o_proj",
                "k_proj",
                "v_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        )

        modelid = "google/paligemma-3b-pt-224"

        self.processor = AutoProcessor.from_pretrained(modelid)

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            modelid,
            torch_dtype=torch.bfloat16,
            device_map=device,
            revision="bfloat16",
            quantization_config=bnb_config,
        )
        # self.model.to(device)
        self.model = get_peft_model(self.model, lora_config)

    def setup(self, stage: str):

        self.train_dataset = load_from_disk(self.train_data_path)
        self.test_dataset = load_from_disk(self.test_data_path)

        self.bertscore_metric = evaluate.load("bertscore")

    def teardown(self, stage: str):

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx):

        input_ids, token_type_ids, attention_mask, pixel_values, captions = batch

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            labels=captions,
        )
        loss = outputs.loss
        perplexity = torch.exp(torch.tensor(loss.item())).item()

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.config.train_batch_size,
            logger=True,
            add_dataloader_idx=False,
        )
        self.log(
            "train_perplexity",
            perplexity,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.config.train_batch_size,
            logger=True,
            add_dataloader_idx=False,
        )

        return loss

    def validation_step(self, batch, batch_idx):

        input_ids, attention_mask, pixel_values, captions = batch

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=self.config.max_tokens,
            )
        predictions = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        bert_f1_scores = []
        bert_precision_scores = []
        bert_recall_scores = []

        for pred, caption in zip(predictions, captions):
            predicted_text = parts[1] if len(parts := pred.split("\n", 1)) > 1 else pred
            predictions = [predicted_text]
            references = [caption]

            # print(f"caption: {caption}")
            # print(f"predicted: {predicted_text}")

            scores = self.bertscore_metric.compute(
                predictions=predictions, references=references, lang="en"
            )
            f1 = scores["f1"][0]
            precision = scores["precision"][0]
            recall = scores["recall"][0]
            bert_f1_scores.append(f1)
            bert_precision_scores.append(precision)
            bert_recall_scores.append(recall)

        self.log(
            "Precision",
            np.mean(bert_precision_scores),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.config.train_batch_size,
            logger=True,
            add_dataloader_idx=False,
        )
        self.log(
            "Recall",
            np.mean(bert_recall_scores),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.config.train_batch_size,
            logger=True,
            add_dataloader_idx=False,
        )
        self.log(
            "F1",
            np.mean(bert_f1_scores),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.config.train_batch_size,
            logger=True,
            add_dataloader_idx=False,
        )

        return np.mean(bert_f1_scores)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.train_learning_rate
        )

        return optimizer

    def train_dataloader(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            collate_fn=partial(
                train_collate_fn, processor=self.processor, device=device
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.train_batch_size,
            collate_fn=partial(
                eval_collate_fn, processor=self.processor, device=device
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the imagecraft model.")
    parser.add_argument("--dataset", type=str, default="flickr")
    parser.add_argument("--dataset_size", type=str, default="30%")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--learning_rate", type=int, default=2e-5)
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--limit_val_batches", type=int, default=5)

    args = parser.parse_args()

    config = ImageCraftConfig
    config.max_tokens = args.max_tokens
    config.train_dataset = args.dataset
    config.train_dataset_size = args.dataset_size
    config.train_epochs = args.epochs
    config.train_max_epochs = args.max_epochs
    config.train_batch_size = args.batch_size
    config.train_learning_rate = args.learning_rate
    config.train_accumulate_grad_batches = args.accumulate_grad_batches
    config.train_gradient_clip_val = args.gradient_clip_val
    config.train_check_val_every_n_epoch = args.check_val_every_n_epoch
    config.train_warmup_steps = args.warmup_steps
    config.train_precision = args.precision
    config.train_num_nodes = args.num_nodes
    config.train_limit_val_batches = args.limit_val_batches

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["USER"] = "imagecraft"

    torch.set_float32_matmul_precision("high")

    dataset = args.dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env_config = tools.load_config()

    model_dir = env_config[f"model_dir"]
    tensorboard_log_dir = env_config["data"]["tensorboard_log_dir"]

    model_flavor = (
        "nsandiman/imagecraft-ft-fk-224"
        if dataset == "flickr"
        else "nsandiman/imagecraft-ft-co-224"
    )

    imageCraft_trainer = ImageCraftTrainer(config)

    trainer = Trainer(
        accelerator="auto",
        strategy="auto",
        max_epochs=config.train_max_epochs,
        accumulate_grad_batches=config.train_accumulate_grad_batches,
        check_val_every_n_epoch=config.train_check_val_every_n_epoch,
        gradient_clip_val=config.train_gradient_clip_val,
        precision=config.train_precision,
        limit_val_batches=config.train_limit_val_batches,
        num_sanity_val_steps=0,
        default_root_dir=model_dir,
        callbacks=[PushToHubCallback(), early_stop_callback],
        logger=TensorBoardLogger(name="imageCraft", save_dir=tensorboard_log_dir),
    )

    trainer.fit(imageCraft_trainer)
