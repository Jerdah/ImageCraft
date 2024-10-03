from argparse import Namespace
from os import PathLike
from pathlib import Path
import os
import time
from typing import Optional, Tuple, Union
from PIL import Image
import torch
from torch import nn
import torchaudio
from src.model import voicecraft
from src.model.modules.gemma import GemmaForCausalLM, KVCache
from src.model.modules.imagecraftconfig import ImageCraftConfig
from src.model.modules.paligemmaprocessor import PaliGemmaProcessor
from src.model.modules.siglip import SiglipVisionModel

from transformers import (
    AutoTokenizer,
    PaliGemmaForConditionalGeneration,
)

from src.model.modules.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
)
from src.utils.model_utils import get_model_inputs
from src.utils.util import (
    replace_numbers_with_words,
    sample_top_p,
    save_to_buffer,
    save_to_file,
    split_line_to_sentences,
)


class ImageCraftMultiModalProjector(nn.Module):
    def __init__(self, config: ImageCraftConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.vision_config.hidden_size,
            config.vision_config.projection_dim,
            bias=True,
        )

    def forward(self, image_features):
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Projection_Dim]
        hidden_states = self.linear(image_features)
        return hidden_states


class ImageCraft(nn.Module):
    config_class = ImageCraftConfig

    def __init__(self, config: ImageCraftConfig):
        # super().__init__(config)
        super(ImageCraft, self).__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = ImageCraftMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size

        self.language_model = GemmaForCausalLM(config.text_config)

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

        self.voicecraft_model = None
        self.audio_tokenizer = None
        self.text_tokenizer = None
        self.processor = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ):
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        # Shape: [Batch_Size, Seq_Len, Hidden_Size]
        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        final_embedding = torch.zeros(
            batch_size,
            sequence_length,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        # Shape: [Batch_Size, Seq_Len]. True for text tokens
        text_mask = (input_ids != self.config.image_token_index) & (
            input_ids != self.pad_token_id
        )
        # Shape: [Batch_Size, Seq_Len]. True for image tokens
        image_mask = input_ids == self.config.image_token_index
        # Shape: [Batch_Size, Seq_Len]. True for padding tokens
        pad_mask = input_ids == self.pad_token_id

        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embeddings
        final_embedding = torch.where(
            text_mask_expanded, inputs_embeds, final_embedding
        )
        # Insert image embeddings. We can't use torch.where because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
        final_embedding = final_embedding.masked_scatter(
            image_mask_expanded, scaled_image_features
        )
        # Zero out padding tokens
        final_embedding = torch.where(
            pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding
        )

        #### CREATE THE ATTENTION MASK ####

        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't need to mask anything, since each query should be able to attend all previous tokens.
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            position_ids = (
                (attention_mask.cumsum(-1))
                .masked_fill_((attention_mask == 0), 1)
                .to(device)
            )

        return final_embedding, causal_mask, position_ids

    def _generate_caption(self, img, max_tokens=100, do_sample=False):
        prompt = "describe the image en"
        try:
            img.verify()
            image = img
        except Exception:
            image = Image.open(img)
        model_inputs = get_model_inputs(self.processor, prompt, image, self.device)
        image.close()
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        pixel_values = model_inputs["pixel_values"]

        kv_cache = KVCache()

        stop_token = self.processor.tokenizer.eos_token_id
        generated_tokens = []

        for _ in range(max_tokens):
            outputs = self(
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
        decoded_text = (
            parts[1] if len(parts := decoded_text.split("\n", 1)) > 1 else decoded_text
        )

        return decoded_text

    def _generate_speech(self, text: str, output_type="file"):

        sentences = split_line_to_sentences(text)

        voice_audio = (
            f"media/voicecraft/voices/{self.config.voicecraft_config.voice_audio_path}"
        )
        voice_transcript = self.config.voicecraft_config.voice_audio_transcript
        cut_off_sec = self.config.voicecraft_config.cut_off_sec

        decode_config = {
            "top_k": self.config.voicecraft_config.top_k,
            "top_p": self.config.voicecraft_config.top_p,
            "temperature": self.config.voicecraft_config.temperature,
            "stop_repetition": self.config.voicecraft_config.stop_repetition,
            "kvcache": self.config.voicecraft_config.kvcache,
            "codec_audio_sr": self.config.voicecraft_config.codec_audio_sr,
            "codec_sr": self.config.voicecraft_config.codec_sr,
            "silence_tokens": self.config.voicecraft_config.silence_tokens,
            "sample_batch_size": self.config.voicecraft_config.sample_batch_size,
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

        output = None

        if output_type == "file":
            output = save_to_file(audio_tensors, decode_config["codec_audio_sr"])
        else:
            output = save_to_buffer(audio_tensors, decode_config["codec_audio_sr"])

        # Empty cuda cache between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

        # Make sure the input is right-padded
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # 1. Extra the input embeddings
        # shape: (Batch_Size, Seq_Len, Hidden_Size)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. Merge text and images
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
        image_features = self.multi_modal_projector(selected_image_feature)

        # Merge the embeddings of the text tokens and the image tokens
        inputs_embeds, attention_mask, position_ids = (
            self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask, kv_cache
            )
        )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs

    @torch.inference_mode()
    def generate(self, image, max_tokens=300, do_sample=False, output_type="file"):
        transcript = self._generate_caption(image, max_tokens, do_sample)
        speech_output = self._generate_speech(transcript, output_type)
        return transcript, speech_output

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, PathLike]],
        *model_args,
        **kwargs,
    ):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        imagecraft_cache_dir = "models/pretrained/imagecraft"
        voicecraft_cache_dir = "models/pretrained/voicecraft"
        Path(imagecraft_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(voicecraft_cache_dir).mkdir(parents=True, exist_ok=True)

        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True,
            cache_dir=imagecraft_cache_dir,
        )

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, padding_side="right"
        )
        assert tokenizer.padding_side == "right"

        config = base_model.config.to_dict()

        config = ImageCraftConfig(**config)

        state_dict = base_model.state_dict()

        model = cls(config).to(device)

        # Load the state dict of the model
        model.load_state_dict(state_dict, strict=False)

        # Tie weights
        model.tie_weights()

        num_image_tokens = model.config.vision_config.num_image_tokens
        image_size = model.config.vision_config.image_size
        model.processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)
        model = model.eval()
        # Load voicecraft module

        model.voicecraft_model = voicecraft.VoiceCraft.from_pretrained(
            f"pyp1/VoiceCraft_{model.config.voicecraft_config.model_name.replace('.pth', '')}",
            cache_dir=voicecraft_cache_dir,
        )

        encodec_fn = f"{voicecraft_cache_dir}/{model.config.voicecraft_config.encodec}"

        if not os.path.exists(encodec_fn):
            os.system(
                f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/{model.config.voicecraft_config.encodec}"
            )
            os.system(f"mv {model.config.voicecraft_config.encodec} {encodec_fn}")

        model.audio_tokenizer = AudioTokenizer(
            signature=encodec_fn,
            device=device,
        )
        model.text_tokenizer = TextTokenizer(backend="espeak")

        model.voicecraft_model.to(device)

        return model
