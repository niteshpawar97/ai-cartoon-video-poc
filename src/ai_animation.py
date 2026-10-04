"""Real character/object movement via AnimateDiff (text-to-video motion module
on top of the same SD1.5 cartoon checkpoint used for stills).

This is the heavier, opt-in alternative to src/animation.py's ffmpeg camera
zoom/pan. It genuinely moves subjects within the scene (walking, running,
reacting) rather than just panning a camera over a static image - but it
needs considerably more VRAM and time. On the 6GB RTX 2060 this project was
primarily developed on, this reliably runs out of memory or is impractically
slow; it is intended for a bigger GPU (e.g. a rented AWS EC2 g4dn.xlarge /
g5.xlarge instance), where 16GB+ VRAM is routinely available.

Not a cloud AI API: AnimateDiff and the base checkpoint are the same local,
open-source models used throughout this project. Running them on a rented
EC2 GPU instead of a laptop GPU is still 100% local/self-hosted inference -
no third-party AI service is called.
"""
import gc
import shutil
import subprocess

from src import config

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

_pipe = None


def load_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe
    import torch
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler

    cache_dir = str(config.MODEL_CACHE_DIR / "huggingface")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # NOTE: do NOT pass variant="fp16" here. guoyww/animatediff-motion-adapter-v1-5-2's
    # "fp16" variant file on the Hub is mislabeled (it actually contains fp32 data
    # internally). Loading it through the variant="fp16" code path silently produces
    # a corrupted model that generates pure noise with no error or warning - confirmed
    # by bisecting through scheduler/VAE-precision/cpu-offload/checkpoint-choice (all
    # ruled out) down to this one flag. Loading the default (non-variant) weights and
    # letting torch_dtype do the fp16 cast works correctly.
    adapter = MotionAdapter.from_pretrained(
        config.ANIMATEDIFF_ADAPTER_ID, cache_dir=cache_dir, torch_dtype=dtype,
    )
    pipe = AnimateDiffPipeline.from_pretrained(
        config.SD_MODEL_ID, motion_adapter=adapter, cache_dir=cache_dir,
        torch_dtype=dtype, safety_checker=None,
    )
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, beta_schedule="linear",
        clip_sample=False, timestep_spacing="linspace", steps_offset=1,
    )
    if torch.cuda.is_available():
        pipe.enable_attention_slicing()
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe = pipe.to("cuda")
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


def generate_scene_clip(
    prompt: str, out_path, seed: int = 42,
    num_frames: int = None, fps: int = None, size: int = None,
):
    """Generate a short animated (real-motion) clip for one scene.

    prompt should describe both the character/style AND the motion/action,
    e.g. "<character description>, walking forward, 2D cartoon style".
    """
    import torch
    from diffusers.utils import export_to_video

    num_frames = num_frames or config.AI_VIDEO_NUM_FRAMES
    fps = fps or config.AI_VIDEO_FPS
    size = size or config.SD_IMAGE_SIZE

    pipe = load_pipeline()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    output = pipe(
        prompt=prompt,
        negative_prompt=config.NEGATIVE_PROMPT,
        num_frames=num_frames,
        guidance_scale=config.AI_VIDEO_GUIDANCE_SCALE,
        num_inference_steps=config.AI_VIDEO_STEPS,
        width=size,
        height=size,
        generator=generator,
    )
    frames = output.frames[0]
    export_to_video(frames, str(out_path), fps=fps)
    return out_path


def normalize_clip(raw_path, out_path, duration: float, size: int = 768, final_fps: int = 25):
    """Resize the raw AnimateDiff clip to the project's standard resolution/fps
    and match it to the scene's target duration, so it concatenates cleanly
    with other scenes downstream (same approach as src/video.py expects).

    If the generated clip is shorter than `duration`, the last frame is held
    (frozen) to fill the remainder rather than looping the action, since
    looping a one-shot action (e.g. "dog grabs umbrella and runs off") would
    jump-cut awkwardly; if longer, it's trimmed.
    """
    vf = f"scale={size}:{size}:flags=lanczos,fps={final_fps}"
    cmd = [
        FFMPEG, "-y", "-i", str(raw_path),
        "-vf", f"{vf},tpad=stop_mode=clone:stop_duration={duration}",
        "-t", f"{duration:.3f}",
        "-pix_fmt", "yuv420p", "-an",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg normalize_clip failed for {raw_path}:\n{result.stderr[-2000:]}")
    return out_path
