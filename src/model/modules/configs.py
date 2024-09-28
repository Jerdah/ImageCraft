from transformers import PretrainedConfig


class ImageCraftConfig:
    def __init__(
        self,
        voicecraft_model_name="gigaHalfLibri330M_TTSEnhanced_max16s.pth",
        voicecraft_encodec="encodec_4cb2048_giga.th",
        voicecraft_top_k=0,
        voicecraft_top_p=0.9,
        voicecraft_temperature=1,
        voicecraft_kvcache=1,
        voicecraft_codec_sr=50,
        voicecraft_codec_audio_sr=16000,
        voicecraft_silence_tokens=[1388, 1898, 131],
        voicecraft_stop_repetition=3,
        voicecraft_sample_batch_size=2,
        voicecraft_seed=1,
        voicecraft_cut_off_sec=67.87,
        voicecraft_voice_audio="84_121550_000074_000000.wav",
        voicecraft_voice_transcript="But when I had approached so near to them The common object, which the sense deceives, Lost not by distance any of its marks",
        max_tokens=100,
        temperature=0.8,
        top_p=0.9,
        train_batch_size=4,
        train_dataset="flickr",
        train_epochs=5,
        train_max_epochs=10,
        train_learning_rate=2e-5,
        train_accumulate_grad_batches=2,
        train_gradient_clip_val=1.0,
        train_check_val_every_n_epoch=1,
        train_warmup_steps=50,
        train_precision="bf16-mixed",
        train_num_nodes=1,
        train_limit_val_batches=5,
    ):
        self.voicecraft_model_name = voicecraft_model_name
        self.voicecraft_encodec = voicecraft_encodec
        self.voicecraft_top_k = voicecraft_top_k
        self.voicecraft_top_p = voicecraft_top_p
        self.voicecraft_temperature = voicecraft_temperature
        self.voicecraft_kvcache = voicecraft_kvcache
        self.voicecraft_codec_sr = voicecraft_codec_sr
        self.voicecraft_codec_audio_sr = voicecraft_codec_audio_sr
        self.voicecraft_silence_tokens = voicecraft_silence_tokens
        self.voicecraft_stop_repetition = voicecraft_stop_repetition
        self.voicecraft_sample_batch_size = voicecraft_sample_batch_size
        self.voicecraft_seed = voicecraft_seed
        self.voicecraft_cut_off_sec = voicecraft_cut_off_sec
        self.voicecraft_voice_audio = voicecraft_voice_audio
        self.voicecraft_voice_transcript = voicecraft_voice_transcript
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.train_batch_size = train_batch_size
        self.train_dataset = train_dataset
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
