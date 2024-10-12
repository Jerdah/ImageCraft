import argparse
from functools import partial
from lightning import LightningModule
import numpy as np
import torch

from datasets import load_from_disk

from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig


from src.data.download import download_dataset
from src.model.modules.imagecraft import ImageCraft
from src.model.modules.trainconfig import TrainConfig
from src.utils import tools

from torch.utils.data import DataLoader

from src.utils.model_utils import get_config
from src.utils.train_utils import eval_collate_fn, save_to_hub, train_collate_fn

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import EarlyStopping


from aac_metrics import evaluate


class ImageCraftModule(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        imagecraft_config = get_config()

        self.model = ImageCraft(imagecraft_config)

    def prepare_data(self):

        env_config = tools.load_config()

        processed_dir = env_config["data"]["processed_dir"]

        train_data_path = f"{processed_dir}/{self.config.train_dataset}/train"
        test_data_path = f"{processed_dir}/{self.config.train_dataset}/test"

        if self.config.train_dataset == "flickr" or self.config.train_dataset == "coco":

            download_dataset(self.config.train_dataset, self.config.train_dataset_size)

            self.training_dataset = load_from_disk(train_data_path)
            self.testing_dataset = load_from_disk(test_data_path)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
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
            device_map="auto",
            revision="bfloat16",
            quantization_config=bnb_config,
        )
        self.model = get_peft_model(self.model, lora_config)

    def setup(self, stage: str):
        pass

    def training_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        attention_mask = batch["attention_mask"]
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            labels=labels,
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

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        cider_scores = []
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
            bleu_1_scores = []
            bleu_2_scores = []
            bleu_3_scores = []
            bleu_4_scores = []
            rouge_l_scores = []
            meteor_scores = []
            cider_d_scores = []
            spice_scores = []
            spider_scores = []

            candidates = []
            references = []

            for pred, captions in zip(predictions, labels):
                # predicted_text = (
                #     parts[1] if len(parts := pred.split("\n", 1)) > 1 else pred
                # )
                predicted_text = pred.replace("caption en", "").replace("\n", "")
                captions = [caption.replace("\n", "") for caption in captions]
                candidates.append(predicted_text)
                references.append(captions)

            corpus_scores = self.calculate_corpus_scores(references, candidates)

            bleu_1_scores.append(corpus_scores["bleu_1"].item())
            bleu_2_scores.append(corpus_scores["bleu_2"].item())
            bleu_3_scores.append(corpus_scores["bleu_3"].item())
            bleu_4_scores.append(corpus_scores["bleu_4"].item())
            rouge_l_scores.append(corpus_scores["rouge_l"].item())
            meteor_scores.append(corpus_scores["meteor"].item())
            cider_d_scores.append(corpus_scores["cider_d"].item())
            spice_scores.append(corpus_scores["spice"].item())
            spider_scores.append(corpus_scores["spider"].item())

        self.log(
            "bleu_1",
            round(np.mean(bleu_1_scores), 2),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.config.train_batch_size,
            logger=True,
            add_dataloader_idx=False,
        )
        self.log(
            "bleu_2",
            round(np.mean(bleu_2_scores), 2),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.config.train_batch_size,
            logger=True,
            add_dataloader_idx=False,
        )
        self.log(
            "bleu_3",
            round(np.mean(bleu_3_scores), 2),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.config.train_batch_size,
            logger=True,
            add_dataloader_idx=False,
        )
        self.log(
            "bleu_4",
            round(np.mean(bleu_4_scores), 2),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.config.train_batch_size,
            logger=True,
            add_dataloader_idx=False,
        )
        self.log(
            "rouge_1",
            round(np.mean(rouge_l_scores), 2),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.config.train_batch_size,
            logger=True,
            add_dataloader_idx=False,
        )
        self.log(
            "meteor",
            round(np.mean(meteor_scores), 2),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.config.train_batch_size,
            logger=True,
            add_dataloader_idx=False,
        )
        self.log(
            "cider_d",
            round(np.mean(cider_d_scores), 2),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.config.train_batch_size,
            logger=True,
            add_dataloader_idx=False,
        )
        self.log(
            "spice",
            round(np.mean(spice_scores), 2),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.config.train_batch_size,
            logger=True,
            add_dataloader_idx=False,
        )
        self.log(
            "spider",
            round(np.mean(spider_scores), 2),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.config.train_batch_size,
            logger=True,
            add_dataloader_idx=False,
        )

        return round(np.mean(cider_d_scores), 2)

    def on_train_end(self):

        repository = (
            "nsandiman/imagecraft-ft-fk-224-pre"
            if self.config.train_dataset == "flickr"
            else "nsandiman/imagecraft-ft-co-224-pre"
        )

        save_to_hub(
            self.model, self.processor.tokenizer, repository, "Final finetuned model"
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.train_learning_rate
        )

        return optimizer

    def train_dataloader(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return DataLoader(
            self.training_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=partial(
                train_collate_fn, processor=self.processor, device=device
            ),
        )

    def val_dataloader(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return DataLoader(
            self.testing_dataset,
            num_workers=0,
            batch_size=self.config.train_batch_size,
            collate_fn=partial(
                eval_collate_fn, processor=self.processor, device=device
            ),
        )

    def calculate_corpus_scores(
        self, references: list[list[str]], candidates: list[str]
    ):
        corpus_scores, _ = evaluate(candidates, references)
        return corpus_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the imagecraft model.")
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--dataset_size", type=str, default="100%")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--learning_rate", type=int, default=2e-5)
    parser.add_argument("--accumulate_grad_batches", type=int, default=2)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--precision", type=str, default="bf16-true")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--limit_val_batches", type=int, default=10)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--log_to", type=str, default="tensorboard")

    args = parser.parse_args()

    env_config = tools.load_config()

    checkpoint_dir = env_config["checkpoint_dir"]
    imagecraft_checkpoint_dir = f"{checkpoint_dir}/imagecraft"

    tensorboard_log_dir = env_config["data"]["tensorboard_log_dir"]
    wandb_log_dir = env_config["data"]["wandb_dir"]

    config = TrainConfig
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
    config.train_log_every_n_steps = args.log_every_n_steps
    config.train_log_to = args.log_to
    config.train_wandb_logger = (
        WandbLogger(name="imagecraft", log_model="all", save_dir=wandb_log_dir)
        if args.log_to == "wandb"
        else None
    )

    dataset = args.dataset

    model = ImageCraftModule(config)
    # model = torch.compile(model)

    tensorboard_logger = TensorBoardLogger(
        name="imagecraft", save_dir=tensorboard_log_dir
    )

    logger = (
        tensorboard_logger
        if config.train_log_to == "tensorboard"
        else config.train_wandb_logger
    )

    trainer = Trainer(
        accelerator="gpu",
        strategy="auto",
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        min_epochs=1,
        max_epochs=config.train_max_epochs,
        accumulate_grad_batches=config.train_accumulate_grad_batches,
        check_val_every_n_epoch=config.train_check_val_every_n_epoch,
        log_every_n_steps=1,
        gradient_clip_val=config.train_gradient_clip_val,
        precision=config.train_precision,
        limit_val_batches=config.train_limit_val_batches,
        num_sanity_val_steps=0,
        default_root_dir=imagecraft_checkpoint_dir,
        callbacks=[
            EarlyStopping(monitor="train_loss", patience=3, verbose=False, mode="min"),
        ],
        logger=logger,
    )

    trainer.fit(model)
