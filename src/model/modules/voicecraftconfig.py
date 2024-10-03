class VoiceCraftConfig:

    def __init__(
        self,
        model_name="330M_TTSEnhanced.pth",  # "gigaHalfLibri330M_TTSEnhanced_max16s.pth",
        encodec="encodec_4cb2048_giga.th",
        top_k=0,
        top_p=0.9,
        temperature=1,
        kvcache=1,
        codec_sr=50,
        codec_audio_sr=16000,
        silence_tokens=[1388, 1898, 131],
        stop_repetition=3,
        sample_batch_size=2,
        seed=1,
        cut_off_sec=7.87,
        voice_audio_path="84_121550_000074_000000.wav",
        voice_audio_transcript="But when I had approached so near to them The common object, which the sense deceives, Lost not by distance any of its marks",
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.encodec = encodec
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.kvcache = kvcache
        self.codec_sr = codec_sr
        self.codec_audio_sr = codec_audio_sr
        self.silence_tokens = silence_tokens
        self.stop_repetition = stop_repetition
        self.sample_batch_size = sample_batch_size
        self.seed = seed
        self.cut_off_sec = cut_off_sec
        self.voice_audio_path = voice_audio_path
        self.voice_audio_transcript = voice_audio_transcript
