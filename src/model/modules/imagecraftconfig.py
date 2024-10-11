# from src.model.modules.gemma import GemmaConfig
# from src.model.modules.siglip import SiglipVisionConfig
from src.model.modules.voicecraftconfig import VoiceCraftConfig

from transformers import SiglipVisionConfig, GemmaConfig, PretrainedConfig


class ImageCraftConfig(PretrainedConfig):

    model_type = "imagecraft"

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        voicecraft_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.is_encoder_decoder = False

        self.pad_token_id = pad_token_id if pad_token_id is not None else -1

        self.vision_config = SiglipVisionConfig(**vision_config)

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (
            self.vision_config.image_size // self.vision_config.patch_size
        ) ** 2
        self.vision_config.projection_dim = projection_dim

        self.voicecraft_config = VoiceCraftConfig(**voicecraft_config)

        super().__init__(**kwargs)
