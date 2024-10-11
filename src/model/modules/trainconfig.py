from transformers import PretrainedConfig


class TrainConfig:
    def __init__(
        self,
        max_tokens=100,
        temperature=0.8,
        top_p=0.9,
        train_batch_size=4,
        train_dataset="flickr",
        train_dataset_size="10%",
        train_epochs=5,
        train_max_epochs=10,
        train_learning_rate=2e-5,
        train_accumulate_grad_batches=4,
        train_gradient_clip_val=1.0,
        train_check_val_every_n_epoch=1,
        train_warmup_steps=2,
        train_precision="bf16-true",
        train_num_nodes=1,
        train_limit_val_batches=5,
        train_log_every_n_steps=50,
        train_log_to="tensorboard",
        train_wandb_logger=None,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.train_batch_size = train_batch_size
        self.train_dataset = train_dataset
        self.train_dataset_size = train_dataset_size
        self.train_epochs = train_epochs
        self.train_max_epochs = train_max_epochs
        self.train_learning_rate = train_learning_rate
        self.train_accumulate_grad_batches = train_accumulate_grad_batches
        self.train_gradient_clip_val = train_gradient_clip_val
        self.train_check_val_every_n_epoch = train_check_val_every_n_epoch
        self.train_warmup_steps = train_warmup_steps
        self.train_precision = train_precision
        self.train_num_nodes = train_num_nodes
        self.train_limit_val_batches = train_limit_val_batches
        self.train_log_every_n_steps = train_log_every_n_steps
        self.train_log_to = train_log_to
        self.train_wandb_logger = train_wandb_logger
