import argparse
from functools import partial
import os
import torch

from datasets import load_from_disk


from transformers import PaliGemmaForConditionalGeneration

from transformers import AutoProcessor

from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig


from src.data.download import download_coco, download_flickr
from src.model.modules.configs import ImageCraftConfig
from src.utils import tools


from src.utils.train_utils import train_collate_fn

from transformers import TrainingArguments
from transformers import Trainer


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

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    repository = (
        "nsandiman/imagecraft-ft-fk-224"
        if config.train_dataset == "flickr"
        else "nsandiman/imagecraft-ft-co-224"
    )

    dataset = "flickr"

    env_config = tools.load_config()

    raw_dir = env_config["data"]["raw_dir"]
    interim_dir = env_config["data"]["interim_dir"]
    processed_dir = env_config["data"]["processed_dir"]
    model_dir = env_config[f"model_dir"]
    train_data_path = f"{processed_dir}/{dataset}/train"
    test_data_path = f"{processed_dir}/{dataset}/test"

    if config.train_dataset == "flickr":
        download_flickr(
            raw_dir,
            interim_dir,
            processed_dir,
            dataset_size=config.train_dataset_size,
        )
    else:
        download_coco(
            raw_dir,
            interim_dir,
            processed_dir,
            dataset_size=config.train_dataset_size,
        )

    training_dataset = load_from_disk(train_data_path)
    testing_dataset = load_from_disk(test_data_path)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
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

    processor = AutoProcessor.from_pretrained(modelid)

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        modelid, device_map=device, quantization_config=bnb_config
    )
    model = get_peft_model(model, lora_config)

    PROMPT = "<image>Describe en<bos>"

    def train_collate_fn(examples):
        images = [example["image"] for example in examples]
        texts = [PROMPT for _ in examples]
        captions = [example["caption"][0] for example in examples]

        tokens = processor(
            text=texts,
            images=images,
            suffix=captions,
            return_tensors="pt",
            padding="longest",
            do_convert_rgb=True,
        ).to(device)

        tokens = tokens.to(torch.bfloat16).to(device)

        return tokens

    train_args = TrainingArguments(
        num_train_epochs=2,
        remove_unused_columns=False,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        learning_rate=2e-5,
        weight_decay=1e-6,
        adam_beta2=0.999,
        logging_steps=100,
        optim="adamw_hf",
        save_strategy="steps",
        save_steps=1000,
        push_to_hub=True,
        save_total_limit=1,
        output_dir=model_dir,
        bf16=True,
        report_to=["tensorboard"],
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=training_dataset,
        data_collator=train_collate_fn,
        tokenizer=processor.tokenizer,
    )

    trainer.train()
    trainer.push_to_hub("nsandiman/imagecraft-ft-fk-224")
