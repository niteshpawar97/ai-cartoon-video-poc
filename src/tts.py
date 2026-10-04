"""Local Hindi TTS using Facebook MMS-TTS (VITS), no cloud API."""
import gc

from src import config

_model = None
_tokenizer = None
_sample_rate = None


def load_tts():
    global _model, _tokenizer, _sample_rate
    if _model is not None:
        return _model, _tokenizer, _sample_rate
    import torch
    from transformers import VitsModel, AutoTokenizer

    cache_dir = str(config.MODEL_CACHE_DIR / "huggingface")
    _tokenizer = AutoTokenizer.from_pretrained(config.TTS_MODEL_ID, cache_dir=cache_dir)
    _model = VitsModel.from_pretrained(config.TTS_MODEL_ID, cache_dir=cache_dir)
    if torch.cuda.is_available():
        _model = _model.to("cuda")
    _model.eval()
    _sample_rate = _model.config.sampling_rate
    return _model, _tokenizer, _sample_rate


def unload_tts():
    global _model, _tokenizer
    if _model is not None:
        import torch
        del _model
        _model = None
        _tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_audio(text: str, out_path) -> float:
    """Synthesize Hindi speech to a WAV file. Returns duration in seconds."""
    import torch
    import soundfile as sf

    model, tokenizer, sample_rate = load_tts()
    inputs = tokenizer(text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs).waveform
    waveform = output.squeeze().detach().cpu().numpy()
    sf.write(str(out_path), waveform, sample_rate)
    duration = len(waveform) / sample_rate
    return duration
