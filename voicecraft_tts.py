import os
import io
import argparse
from argparse import Namespace
from models import voicecraft
import torch
import torchaudio
import tqdm
import uuid
from inference_tts_scale import inference_one_sample
from data.voicecraft.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)
from huggingface_hub import hf_hub_download


import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")
import re
from num2words import num2words


def replace_numbers_with_words(self, sentence):
    sentence = re.sub(r"(\d+)", r" \1 ", sentence)

    def replace_with_words(match):
        num = match.group(0)
        try:
            return num2words(num)
        except:
            return num

    return re.sub(r"\b\d+\b", replace_with_words, sentence)


def get_output_audio(self, audio_tensors, codec_audio_sr):

    result = torch.cat(audio_tensors, 1)
    buffer = io.BytesIO()
    torchaudio.save(buffer, result, int(codec_audio_sr), format="wav")
    buffer.seek(0)
    return buffer.read()


class VoiceCraftTTS:
    """
    A pipeline for converting text to speech.
    """

    def __init__(
        self,
        model_name="gigaHalfLibri330M_TTSEnhanced_max16s.pth",
        encodec_fn="encodec_4cb2048_giga.th",
    ):
        model = voicecraft.VoiceCraft.from_pretrained(
            f"pyp1/VoiceCraft_{model_name.replace('.pth', '')}"
        )

        encodec_path = f"pretrained_models/{encodec_fn}"

        if not os.path.exists(encodec_path):
            os.system(
                f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th"
            )
            os.system(
                f"mv encodec_4cb2048_giga.th ./pretrained_models/encodec_4cb2048_giga.th"
            )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.audio_tokenizer = AudioTokenizer(
            signature=encodec_path, device=self.device
        )
        self.text_tokenizer = TextTokenizer(backend="espeak")

        self.phn2num = model.args.phn2num
        self.config = vars(model.args)
        model.to(self.device)
        self.model = model

        self.orig_audio = "resources/voices/84_121550_000074_000000.wav"
        self.orig_transcript = "But when I had approached so near to them The common object, which the sense deceives, Lost not by distance any of its marks"
        self.cut_off_sec = 67.87

        self.codec_audio_sr = 16000
        self.codec_sr = 50
        self.top_k = 0
        self.top_p = 0.9
        self.temperature = 1
        self.silence_tokens = [1388, 1898, 131]
        self.kvcache = 1
        self.stop_repetition = 3
        self.sample_batch_size = 2
        self.seed = 1

    def generate(self, text):

        text = replace_numbers_with_words(text).replace("  ", " ").replace("  ", " ")

        sentences = sent_tokenize(text.replace("\n", " "))

        info = torchaudio.info(self.orig_audio)
        audio_dur = info.num_frames / info.sample_rate

        audio_tensors = []
        transcript = ""

        for sentence in tqdm(sentences):
            decode_config = {
                "top_k": self.top_k,
                "top_p": self.top_p,
                "temperature": self.temperature,
                "stop_repetition": self.stop_repetition,
                "kvcache": self.kvcache,
                "codec_audio_sr": self.codec_audio_sr,
                "codec_sr": self.codec_sr,
                "silence_tokens": self.silence_tokens,
                "sample_batch_size": self.sample_batch_size,
            }
            transcript = self.orig_transcript
            transcript += sentence + "\n"

            prompt_end_frame = int(min(audio_dur, self.cut_off_sec) * info.sample_rate)

            transcript = (
                replace_numbers_with_words(transcript)
                .replace("  ", " ")
                .replace("  ", " ")
            )

            _, gen_audio = inference_one_sample(
                self.model,
                Namespace(**self.config),
                self.phn2num,
                self.text_tokenizer,
                self.audio_tokenizer,
                self.orig_audio,
                transcript,
                self.device,
                decode_config,
                prompt_end_frame,
            )
            gen_audio = gen_audio[0].cpu()
            audio_tensors.append(gen_audio)

        output_audio = get_output_audio(audio_tensors, self.codec_audio_sr)
        return output_audio


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train an image captioning model.")
    parser.add_argument(
        "--text", type=str, default="Image craft is an image to speech engine."
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--model", type=str, default="gigaHalfLibri330M_TTSEnhanced_max16s.pth"
    )
    parser.add_argument("--encodec", type=str, default="encodec_4cb2048_giga.th")

    args = parser.parse_args()

    pipeline = VoiceCraftTTS(args.model, args.encodec)
    audio_buffer = pipeline.generate(args.text)
    unique_filename = str(uuid.uuid4())
    audio_path = f"resources/generated/{unique_filename}.wav"
    with open(audio_path, "wb") as f:
        f.write(audio_buffer.getbuffer())
    print(audio_path)
