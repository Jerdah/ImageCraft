import torch
from torchvision import transforms
from PIL import Image
from data.tokenizer import Tokenizer
from data import config
from models.modules.imagecaptioner import ImageCaptioner
from voicecraft_tts_pipeline import VoiceCraftTTSPipeline


class ImageCraft:
    """
    The imagecraft main class.
    """

    def __init__(self, checkpoint: str, max_len: int, device: str, tokenizer=None):

        if tokenizer is None:
            # Initialize the tokenizer if not provided
            self.tokenizer = Tokenizer(
                tokenizer_name="gpt2",
                special_tokens_dict={
                    "bos_token": "[BOS]",
                    "eos_token": "[EOS]",
                    "pad_token": "[PAD]",
                },
            )
        else:
            self.tokenizer = tokenizer

        # Update model configuration with tokenizer-specific settings
        config["gpt_kwargs"]["vocab_size"] = self.tokenizer.vocab_size
        config["gpt_kwargs"]["ignore_index"] = self.tokenizer.get_vocab()[
            self.tokenizer.pad_token
        ]
        self.max_len = max_len
        self.device = device

        self.model = ImageCaptioner(config).from_pretrained(checkpoint, device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(size=(config["img_size"], config["img_size"])),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def generate(self, image_path: str) -> str:

        # Load and transform the image
        image_tensor = (
            self.transform(Image.open(image_path)).unsqueeze(0).to(self.device)
        )

        # Generate caption using the model
        tokens = self.model.generate(
            image_tensor,
            sos_token=self.tokenizer.get_vocab()["[BOS]"],
            eos_token=self.tokenizer.get_vocab()["[EOS]"],
            max_len=self.max_len,
        )

        # Decode the generated token IDs to a caption string
        decoded_caption = self.tokenizer.decode(
            token_ids=[token.item() for token in tokens[1:-1]]
        )
        voicecraft_pipeline = VoiceCraftTTSPipeline()
        return voicecraft_pipeline.generate(decoded_caption)
