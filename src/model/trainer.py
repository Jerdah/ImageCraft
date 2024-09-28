import argparse
from functools import partial
import os
from lightning import LightningModule
import torch

from datasets import load_from_disk

from pytorch_lightning.callbacks import EarlyStopping


from transformers import PaliGemmaForConditionalGeneration

from transformers import AutoProcessor

from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig


from src.model.modules.configs import ImageCraftConfig
from src.utils import tools

from torch.utils.data import DataLoader

from src.utils.train_utils import eval_collate_fn, train_collate_fn

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger


from huggingface_hub import HfApi

api = HfApi()

model_flavor = "nsandiman/imagecraft-ft-fk-224"


class PushToHubCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.model.push_to_hub(
            model_flavor,
            commit_message=f"Training in progress, epoch {trainer.current_epoch}",
        )

    def on_train_end(self, trainer, pl_module):
        pl_module.processor.push_to_hub(model_flavor, commit_message=f"Training done")
        pl_module.model.push_to_hub(model_flavor, commit_message=f"Training done")


early_stop_callback = EarlyStopping(
    monitor="val_loss", patience=3, verbose=False, mode="min"
)


class ImageCraftTrainer(LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.batch_size = config.train_batch_size

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
            batch_size=self.batch_size,
            logger=True,
        )
        self.log(
            "train_perplexity",
            perplexity,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):

        (
            input_ids,
            token_type_ids,
            attention_mask,
            pixel_values,
            captions,
            reference_captions,
        ) = batch

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            labels=captions,
        )
        loss = outputs.loss

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
            logger=True,
        )

        # generated_ids = self.model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     pixel_values=pixel_values,
        #     max_new_tokens=100,
        # )

        # predictions = self.processor.batch_decode(
        #     generated_ids, skip_special_tokens=True
        # )

        # ed_scores = []
        # bleu_scores = []
        # bleu_weights = (0.25, 0.25, 0, 0)
        # for pred, caption, reference_captions in zip(
        #     predictions, captions, reference_captions
        # ):

        #     bleu_candidate = pred.lower().split()
        #     bleu_reference = [sub.lower().split() for sub in reference_captions]

        #     bleu_score = sentence_bleu(
        #         bleu_reference, bleu_candidate, weights=bleu_weights
        #     )
        #     bleu_score_str = "{:.2f}".format(mean(bleu_score))
        #     bleu_scores.append(bleu_score)

        #     pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
        #     ed_scores.append(
        #         edit_distance(pred, caption) / max(len(pred), len(caption))
        #     )
        #     if len(ed_scores) == 1:
        #         print(f"Prediction: {pred}")
        #         print(f"    Answer: {caption}")
        #         print(f" Normed ED: {ed_scores[0]}")
        #         print(f"Bleu Score: {bleu_score_str}")

        # self.log(
        #     "val_bleu_score",
        #     mean(bleu_scores),
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        #     batch_size=self.batch_size,
        # )
        # self.log(
        #     "val_edit_distance",
        #     mean(ed_scores),
        #     on_step=True,
        #     on_epoch=True,
        #     prog_bar=True,
        #     batch_size=self.batch_size,
        # )

        # return ed_scores  # loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.train_learning_rate
        )

        return optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the imagecraft model.")
    parser.add_argument("--dataset", type=str, default="flickr")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=int, default=2e-5)
    parser.add_argument("--accumulate_grad_batches", type=int, default=2)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--limit_val_batches", type=int, default=5)

    args = parser.parse_args()

    config = ImageCraftConfig
    config.train_dataset = args.dataset
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
    os.environ["USER"] = "nsandiman"

    torch.set_float32_matmul_precision("high")

    dataset = args.dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env_config = tools.load_config()

    model_dir = env_config[f"model_dir"]
    model_path = env_config[f"model_{dataset}"]
    model_checkpoints_path = env_config["model_checkpoints"]
    processed_data_dir = env_config["data"]["processed_dir"]
    log_data_dir = env_config["data"]["log_dir"]
    tensorboard_dir = env_config["data"]["tensorboard_dir"]

    finetuned_model_path = env_config[f"model_{dataset}"]

    train_data_path = f"{processed_data_dir}/{dataset}/train"
    test_data_path = f"{processed_data_dir}/{dataset}/test"

    train_dataset = load_from_disk(train_data_path)
    test_dataset = load_from_disk(test_data_path)

    model_flavor = (
        "nsandiman/imagecraft-ft-fk-224"
        if dataset == "flickr"
        else "nsandiman/imagecraft-ft-co-224"
    )

    # use this for Q-LoRa
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

    # Load the base model and apply LoRA
    modelid = "google/paligemma-3b-pt-224"

    processor = AutoProcessor.from_pretrained(modelid)

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        modelid,
        torch_dtype=torch.bfloat16,
        device_map=device,
        quantization_config=bnb_config,
    )
    model = get_peft_model(model, lora_config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=partial(train_collate_fn, processor=processor, device=device),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train_batch_size,
        shuffle=False,
        collate_fn=partial(eval_collate_fn, processor=processor, device=device),
    )

    imageCraft_trainer = ImageCraftTrainer(config, processor, model)
    imageCraft_trainer.save_hyperparameters()

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
        logger=TensorBoardLogger(save_dir=tensorboard_dir),
    )

    trainer.fit(
        imageCraft_trainer, train_dataloaders=train_loader, val_dataloaders=test_loader
    )
