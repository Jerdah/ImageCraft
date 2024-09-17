import os
from data.dataset_util import train_test_split
from data.tokenizer import Tokenizer
import torch
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import argparse
import time
import warnings
import logging

from data import config
from torch.utils.tensorboard import SummaryWriter


from models.modules.imagecaptioner import ImageCaptioner


def clear_console():
    """
    Clears the console based on the operating system.
    """
    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")


class Trainer:
    """
    Trainer class for image captioning model.
    """

    def __init__(self, model_config, train_config, dls, tokenizer) -> None:

        self.device = train_config["device"]

        # Load model from checkpoint if provided, else initialize a new model
        if train_config["checkpoint"] is not None:
            self.model = ImageCaptioner(model_config).from_pretrained(
                train_config["checkpoint"], self.device
            )
        else:
            self.model = ImageCaptioner(model_config).to(self.device)

        self.train_config = train_config
        self.model_config = model_config
        self.train_dl, self.test_dl = dls
        self.metrics = pd.DataFrame(
            columns=[
                "epoch",
                "train_loss",
                "test_loss",
                "train_perplexity",
                "test_perplexity",
                "elapsed_time",
            ]
        )
        self.tokenizer = tokenizer
        self.writer = SummaryWriter(train_config["experiment_name"])

        # Transformation pipeline for images
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    size=(model_config["img_size"], model_config["img_size"])
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def fit(self, verbose=True):

        global_step = 0
        start_time = time.time()

        # Freeze the pretrained ViT (Vision Transformer) parameters during initial epochs
        if self.model.is_vit_pretrained and self.train_config["freeze_epochs"] > 0:
            for p in self.model.vit.parameters():
                p.requires_grad = False

        # Set up optimizer with learning rates for different model parts
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.model.vit.parameters(),
                    "lr": 0,
                },  # Vit params are initially frozen
                {
                    "params": self.model.dimension_mapping_layer.parameters(),
                    "lr": self.train_config["lr"],
                },
                {"params": self.model.gpt.parameters(), "lr": self.train_config["lr"]},
            ],
            weight_decay=self.train_config["weight_decay"],
        )

        # Initial training with frozen ViT parameters
        for epoch in range(self.train_config["freeze_epochs"]):
            train_loss, train_perplexity, global_step = self._train(epoch, global_step)
            test_loss, test_perplexity, global_step = self._eval(epoch, global_step)
            elapsed_time = time.time() - start_time
            new_row = pd.DataFrame(
                data={
                    "epoch": [epoch + 1],
                    "train_loss": [train_loss],
                    "test_loss": [test_loss],
                    "elapsed_time": [elapsed_time],
                    "train_perplexity": [train_perplexity],
                    "test_perplexity": [test_perplexity],
                }
            )

            # Store training metrics
            self.metrics = pd.concat([self.metrics, new_row], axis=0, ignore_index=True)

            # Clear console and print metrics
            clear_console()
            print(self.metrics.to_string(index=False))

        # Unfreeze the ViT parameters after initial epochs
        if self.model.is_vit_pretrained and self.train_config["freeze_epochs"] > 0:
            for p in self.model.vit.parameters():
                p.requires_grad = True

        self.optimizer.param_groups[0]["lr"] = self.train_config[
            "lr"
        ]  # Unfreeze ViT params

        # Further training with all parameters unfrozen
        for epoch in range(
            self.train_config["freeze_epochs"], self.train_config["epochs"]
        ):
            train_loss, train_perplexity, global_step = self._train(epoch, global_step)
            test_loss, test_perplexity, global_step = self._eval(epoch, global_step)
            elapsed_time = time.time() - start_time
            new_row = pd.DataFrame(
                data={
                    "epoch": [epoch + 1],
                    "train_loss": [train_loss],
                    "test_loss": [test_loss],
                    "elapsed_time": [elapsed_time],
                    "train_perplexity": [train_perplexity],
                    "test_perplexity": [test_perplexity],
                }
            )

            # Store training metrics
            self.metrics = pd.concat([self.metrics, new_row], axis=0, ignore_index=True)

            # Clear console and print metrics
            clear_console()
            print(self.metrics.to_string(index=False))

        # Save the final model checkpoint
        self.save("temp/imagecraft.pt")
        return self.metrics

    def _train(self, epoch, global_step):

        self.model.train()
        total_loss = 0
        train_batchiter = tqdm(
            self.train_dl, desc=f"Processing Training Epoch {epoch:02d}"
        )

        for image, tokens, attn_mask in train_batchiter:
            # Prepare inputs and targets
            input_tokens, target_tokens = tokens[:, :-1], tokens[:, 1:]
            attn_mask = attn_mask[:, :-1]
            image, input_tokens, target_tokens, attn_mask = (
                image.to(self.device),
                input_tokens.to(self.device),
                target_tokens.to(self.device),
                attn_mask.to(self.device),
            )

            # Forward pass and compute loss
            _, loss = self.model(image, input_tokens, attn_mask, target_tokens)
            total_loss += loss.item()

            # Update progress bar with loss and perplexity
            train_batchiter.set_postfix(
                {
                    "Train Loss": f"{loss.item():6.3f}",
                    "Train Perplexity": f"{torch.exp(torch.tensor(loss.item())).item()}",
                }
            )

            # Log the Loss to TensorBoard
            self.writer.add_scalar("Loss/train", loss.item(), global_step)
            self.writer.flush()

            # Backpropagation and optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            global_step += 1

        # Compute average loss and perplexity
        avg_loss = total_loss / len(self.train_dl)
        train_perplexity = torch.exp(torch.tensor(avg_loss))
        return avg_loss, train_perplexity.item(), global_step

    def _eval(self, epoch, global_step):

        self.model.eval()
        total_loss = 0
        test_batchiter = tqdm(self.test_dl, desc=f"Processing Eval Epoch {epoch:02d}")

        with torch.no_grad():
            for image, tokens, attn_mask in test_batchiter:
                # Prepare inputs and targets
                input_tokens, target_tokens = tokens[:, :-1], tokens[:, 1:]
                attn_mask = attn_mask[:, :-1]
                image, input_tokens, target_tokens, attn_mask = (
                    image.to(self.device),
                    input_tokens.to(self.device),
                    target_tokens.to(self.device),
                    attn_mask.to(self.device),
                )

                # Forward pass and compute loss
                _, loss = self.model(image, input_tokens, attn_mask, target_tokens)
                total_loss += loss.item()

                # Update progress bar with loss and perplexity
                test_batchiter.set_postfix(
                    {
                        "Test Loss": f"{loss.item():6.3f}",
                        "Test Perplexity": f"{torch.exp(torch.tensor(loss.item())).item()}",
                    }
                )

                # Log the Loss to TensorBoard
                self.writer.add_scalar("Loss/eval", loss.item(), global_step)
                self.writer.flush()

                global_step += 1

        # Compute average loss and perplexity
        avg_loss = total_loss / len(self.test_dl)
        test_perplexity = torch.exp(torch.tensor(avg_loss))
        return avg_loss, test_perplexity.item(), global_step

    def inference(self, image_path, max_len) -> str:

        # Preprocess image
        image_tensor = (
            self.transform(Image.open(image_path)).unsqueeze(0).to(self.device)
        )

        # Generate tokens using the model
        tokens = self.model.generate(
            image_tensor,
            sos_token=self.tokenizer.get_vocab()["[BOS]"],
            eos_token=self.tokenizer.get_vocab()["[EOS]"],
            max_len=max_len,
        )

        # Decode tokens to generate caption
        return self.tokenizer.decode(token_ids=[token.item() for token in tokens])

    def save(self, file_path):

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_config": self.train_config,
            "model_config": self.model_config,
        }

        torch.save(checkpoint, file_path)

    def plot_metrics(self):
        """
        Plots and saves training and test loss and perplexity metrics.
        """
        plt.figure(figsize=(12, 6))

        # Plot training and test loss
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics["epoch"], self.metrics["train_loss"], label="Train Loss")
        plt.plot(self.metrics["epoch"], self.metrics["test_loss"], label="Test Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Training and Test Loss Over Epochs")
        plt.legend()
        plt.grid(True)

        # Plot training and test perplexity
        plt.subplot(1, 2, 2)
        plt.plot(
            self.metrics["epoch"],
            self.metrics["train_perplexity"],
            label="Train Perplexity",
        )
        plt.plot(
            self.metrics["epoch"],
            self.metrics["test_perplexity"],
            label="Test Perplexity",
        )
        plt.xlabel("epoch")
        plt.ylabel("perplexity")
        plt.title("Training and Test Perplexity Over Epochs")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("metrics.png")
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train an image captioning model.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--freeze_epochs", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--train_size", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    assert args.freeze_epochs <= args.epochs, ValueError(
        f"'freeze_epochs': {args.freeze_epochs} must be <= 'epochs': {args.epochs}"
    )

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Load preprocessed data
    logger.info("Processing data...")
    data_path = "preprocessed_data.csv"
    data = pd.read_csv(data_path)
    logger.info("Data processing complete.")

    # Initialize tokenizer
    tokenizer = Tokenizer(
        tokenizer_name="gpt2",
        special_tokens_dict={
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "pad_token": "[PAD]",
        },
    )

    img_size = 224

    # Set training configuration
    train_config = {
        "epochs": args.epochs,
        "freeze_epochs": args.freeze_epochs,
        "lr": args.lr,
        "device": args.device,
        "weight_decay": 1e-6,
        "experiment_name": "runs/tmodel",
        "checkpoint": None,
        "train_size": args.train_size,
        "batch_size": args.batch_size,
    }

    # Update model configuration with device and tokenizer information
    config["device"] = args.device
    config["img_size"] = img_size
    config["gpt_kwargs"]["device"] = args.device
    config["vit_kwargs"]["device"] = args.device
    config["gpt_kwargs"]["vocab_size"] = tokenizer.vocab_size
    config["vit_kwargs"]["pretrained_model_name"] = "vit_tiny_patch16_224"
    config["gpt_kwargs"]["ignore_index"] = tokenizer.get_vocab()[tokenizer.pad_token]

    # Prepare data loaders for training and testing
    logger.info("Preparing data...")
    train_dl, test_dl = train_test_split(
        train_config=train_config, model_config=config, data=data, tokenizer=tokenizer
    )
    logger.info("Data preparation complete.")

    # Initialize trainer
    trainer = Trainer(
        model_config=config,
        train_config=train_config,
        dls=(train_dl, test_dl),
        tokenizer=tokenizer,
    )

    # Train the model and save metrics
    logger.info("Starting the training process...")
    metrics = trainer.fit()
    metrics.to_csv("metrics.csv")

    # Plot training and test metrics
    trainer.plot_metrics()
    logger.info("Training process completed")
