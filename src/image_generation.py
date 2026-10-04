"""Local cartoon image generation using Stable Diffusion 1.5 (cartoon-finetuned)."""
import gc

from src import config

_pipe = None


def load_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        config.SD_MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
        cache_dir=str(config.MODEL_CACHE_DIR / "huggingface"),
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
    _pipe = pipe
    return _pipe


def unload_pipeline():
    global _pipe
    if _pipe is not None:
        import torch
        del _pipe
        _pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_image(prompt: str, out_path, seed: int = 42, steps: int = 25, size: int = None):
    import torch
    size = size or config.SD_IMAGE_SIZE
    pipe = load_pipeline()
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    image = pipe(
        prompt=prompt,
        negative_prompt=config.NEGATIVE_PROMPT,
        num_inference_steps=steps,
        guidance_scale=7.5,
        width=size,
        height=size,
        generator=generator,
    ).images[0]
    image.save(out_path)
    return out_path
