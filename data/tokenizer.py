# cp from https://github.com/pratyakshagarwal/Image-Caption-Model-Using-Vision-transformer-and-decoder/blob/master/tokenizer.py


import torch
import transformers
import pickle
from typing import Dict
import warnings

class Tokenizer:

    def __init__(self, tokenizer_name, special_tokens_dict=None) -> None:

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        if special_tokens_dict is None:
           warnings.warn(f"'special_tokens_dict' has not been set, using default special_tokens_dict")
           self.tokenizer.add_special_tokens({
               "bos_token": "[BOS]",
               "eos_token": "[EOS]",
               "pad_token": "[PAD]"
           }) 
           self.vocab_size = self.tokenizer.vocab_size + 3
           self.pad_token = '[PAD]'
        else:
            assert 'pad_token' in special_tokens_dict, ValueError("'pad_token' key must be present in the 'special_tokens_dict' passed")                
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.vocab_size = self.tokenizer.vocab_size + len(special_tokens_dict)
            self.pad_token = special_tokens_dict['pad_token']

    def encode(self, text, max_len, padding=True) -> Dict[str, torch.Tensor]:
        return self.tokenizer(text, max_length=max_len, padding='max_length' if padding else True, 
                              return_tensors='pt')

    def decode(self, token_ids) -> str:
        return self.tokenizer.decode(token_ids)
    
    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def get_vocab(self):
        return self.tokenizer.get_vocab()
    
    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)