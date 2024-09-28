import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class ImageCraftOutput:
    loss: Optional[torch.FloatTensor] = None
    caption_ids: Optional[torch.LongTensor] = None
    logits: Optional[torch.FloatTensor] = None
