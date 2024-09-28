from argparse import Namespace
import os
import re
import time
import torch
import torchaudio

import torch.nn as nn


from src.model import voicecraft
from src.model.modules.gemma import KVCache
from src.model.modules.paligemmaprocessor import PaliGemmaProcessor
from src.model.modules.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
)
from src.utils.model_utils import get_model_inputs, load_hf_model
from src.utils.util import (
    get_output_audio,
    replace_numbers_with_words,
    sample_top_p,
    save_output_audio,
    seed_everything,
    split_line_to_sentences,
)


class ImageCraftModel(nn.Module):
    """
    Main imagecraft model class.
    """

    def __init__(self, config) -> None:

        super(ImageCraftModel, self).__init__()

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seed_everything(1)

    def generate(self, image_path: str, prompt: str, max_tokens=100, do_sample=False):

        model_inputs = get_model_inputs(self.processor, prompt, image_path, self.device)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        pixel_values = model_inputs["pixel_values"]

        kv_cache = KVCache()

        stop_token = self.processor.tokenizer.eos_token_id
        generated_tokens = []

        for _ in range(max_tokens):
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
            kv_cache = outputs["kv_cache"]
            next_token_logits = outputs["logits"][:, -1, :]
            if do_sample:
                next_token_logits = torch.softmax(
                    next_token_logits / self.config.temperature, dim=-1
                )
                next_token = sample_top_p(next_token_logits, self.config.top_p)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            assert next_token.size() == (1, 1)
            next_token = next_token.squeeze(0)
            generated_tokens.append(next_token)
            if next_token.item() == stop_token:
                break
            input_ids = next_token.unsqueeze(-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
            )

        generated_tokens = torch.cat(generated_tokens, dim=-1)
        decoded_text = self.processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
        print(decoded_text)
        # voicecraft sampling

        sentences = split_line_to_sentences(decoded_text)

        voice_audio = f"media/voicecraft/voices/{self.config.voicecraft_voice_audio}"
        voice_transcript = self.config.voicecraft_voice_transcript
        cut_off_sec = self.config.voicecraft_cut_off_sec

        decode_config = {
            "top_k": self.config.voicecraft_top_k,
            "top_p": self.config.voicecraft_top_p,
            "temperature": self.config.voicecraft_temperature,
            "stop_repetition": self.config.voicecraft_stop_repetition,
            "kvcache": self.config.voicecraft_kvcache,
            "codec_audio_sr": self.config.voicecraft_codec_audio_sr,
            "codec_sr": self.config.voicecraft_codec_sr,
            "silence_tokens": self.config.voicecraft_silence_tokens,
            "sample_batch_size": self.config.voicecraft_sample_batch_size,
        }

        info = torchaudio.info(voice_audio)
        audio_dur = info.num_frames / info.sample_rate
        prompt_end_frame = int(min(audio_dur, cut_off_sec) * info.sample_rate)

        audio_tensors = []
        transcript = voice_transcript

        for sentence in sentences:

            transcript += sentence + "\n"
            transcript = replace_numbers_with_words(transcript).replace("  ", " ")

            # phonemize
            phn2num = self.voicecraft_model.args.phn2num
            text_tokens = [
                phn2num[phn]
                for phn in tokenize_text(self.text_tokenizer, text=transcript.strip())
                if phn in phn2num
            ]
            text_tokens = torch.LongTensor(text_tokens).unsqueeze(0)
            text_tokens_lens = torch.LongTensor([text_tokens.shape[-1]])

            # encode audio
            encoded_frames = tokenize_audio(
                self.audio_tokenizer,
                voice_audio,
                offset=0,
                num_frames=prompt_end_frame,
            )
            original_audio = encoded_frames[0][0].transpose(2, 1)  # [1,T,K]
            model_args = vars(self.voicecraft_model.args)
            model_args = Namespace(**model_args)

            assert (
                original_audio.ndim == 3
                and original_audio.shape[0] == 1
                and original_audio.shape[2] == model_args.n_codebooks
            ), original_audio.shape

            # forward
            stime = time.time()
            if decode_config["sample_batch_size"] <= 1:
                _, gen_frames = self.voicecraft_model.inference_tts(
                    text_tokens.to(self.device),
                    text_tokens_lens.to(self.device),
                    original_audio[..., : model_args.n_codebooks].to(
                        self.device
                    ),  # [1,T,8]
                    top_k=decode_config["top_k"],
                    top_p=decode_config["top_p"],
                    temperature=decode_config["temperature"],
                    stop_repetition=decode_config["stop_repetition"],
                    kvcache=decode_config["kvcache"],
                    silence_tokens=(
                        eval(decode_config["silence_tokens"])
                        if type(decode_config["silence_tokens"]) == str
                        else decode_config["silence_tokens"]
                    ),
                )  # output is [1,K,T]
            else:
                _, gen_frames = self.voicecraft_model.inference_tts_batch(
                    text_tokens.to(self.device),
                    text_tokens_lens.to(self.device),
                    original_audio[..., : model_args.n_codebooks].to(
                        self.device
                    ),  # [1,T,8]
                    top_k=decode_config["top_k"],
                    top_p=decode_config["top_p"],
                    temperature=decode_config["temperature"],
                    stop_repetition=decode_config["stop_repetition"],
                    kvcache=decode_config["kvcache"],
                    batch_size=decode_config["sample_batch_size"],
                    silence_tokens=(
                        eval(decode_config["silence_tokens"])
                        if type(decode_config["silence_tokens"]) == str
                        else decode_config["silence_tokens"]
                    ),
                )  # output is [1,K,T]
            gen_sample = self.audio_tokenizer.decode([(gen_frames, None)])
            gen_audio = gen_sample[0].cpu()
            audio_tensors.append(gen_audio)

        output_audio = get_output_audio(audio_tensors, decode_config["codec_audio_sr"])

        # Empty cuda cache between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_audio

    def from_pretrained(self, model_path):

        self.model, self.tokenizer = load_hf_model(model_path, self.device)
        self.model = self.model.to(self.device).eval()

        num_image_tokens = self.model.config.vision_config.num_image_tokens
        image_size = self.model.config.vision_config.image_size
        self.processor = PaliGemmaProcessor(
            self.tokenizer, num_image_tokens, image_size
        )

        # Load voicecraft module

        self.voicecraft_model = voicecraft.VoiceCraft.from_pretrained(
            f"pyp1/VoiceCraft_{self.config.voicecraft_model_name.replace('.pth', '')}"
        )

        if not os.path.exists(self.config.voicecraft_encodec):
            os.system(
                f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/{self.config.voicecraft_encodec}"
            )

        self.audio_tokenizer = AudioTokenizer(
            signature=self.config.voicecraft_encodec,
            device=self.device,
        )
        self.text_tokenizer = TextTokenizer(backend="espeak")

        self.voicecraft_model.to(self.device)
