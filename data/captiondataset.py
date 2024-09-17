import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from typing import Tuple

class CaptionDataset(Dataset):
    
    def __init__(self, dataframe: pd.DataFrame, image_size: int, context_length: int, tokenizer) -> None:

        assert dataframe.columns[0] == 'image_name', ValueError("The first column should be the path to the image")
        assert dataframe.columns[1] == "caption", ValueError("The second column should be named 'caption'")

        self.tokenizer = tokenizer
        self.context_length = context_length
        self.df = dataframe
        
        # Transformation pipeline for images
        self.transform = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:

        return self.df.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        image, text = Image.open(self.df.iloc[idx, 0]), self.df.iloc[idx, 1]
        image_tensor = self.transform(image)  # Apply transformations to the image
        op = self.tokenizer(text, max_len=self.context_length + 1)  # Tokenize the caption
        tokens, attention_mask = op['input_ids'].squeeze(), op['attention_mask'].squeeze()
        return image_tensor, tokens, attention_mask