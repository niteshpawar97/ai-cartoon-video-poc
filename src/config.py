"""Central configuration: paths, cache locations, model choices.

Must be imported before torch/transformers/diffusers/huggingface_hub so the
cache-directory environment variables take effect (they are read at import
time by those libraries).
"""
import os
from pathlib import Path

# Keep large model downloads off the small project drive (A:).
MODEL_CACHE_DIR = Path(os.environ.get("AI_CARTOON_MODEL_CACHE", "F:/ai-models-cache"))
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(MODEL_CACHE_DIR / "torch"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
# The "Xet" CDN backend (hf-xet) was observed to stall indefinitely on large
# (multi-GB) file downloads on this network, while classic HTTP downloads of
# the same files succeeded immediately. Force classic HTTP for reliability.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

# ---- Model choices (selected for a 6GB-class NVIDIA GPU) ----------------
# NOTE: llama-cpp-python requires a C/C++ compiler (nmake/cl.exe via MSVC build
# tools) to build on this machine, which is not installed. Rather than pull in a
# multi-GB Visual Studio Build Tools install just for GGUF inference, the LLM
# stage uses the plain transformers safetensors checkpoint instead (already a
# project dependency, no compiler needed). Runs briefly on GPU (fp16, ~3GB VRAM)
# then is unloaded before the image-generation stage claims the VRAM.
LLM_REPO = "Qwen/Qwen2.5-3B-Instruct"

SD_MODEL_ID = "nitrosocke/mo-di-diffusion"  # SD1.5-based, modern cartoon/Disney style
SD_IMAGE_SIZE = 512

TTS_MODEL_ID = "facebook/mms-tts-hin"  # lightweight VITS Hindi TTS

CHARACTER_DESCRIPTION = (
    "small orange cartoon cat, large expressive eyes, blue collar, "
    "funny face, 2D animated cartoon style, modern disney style, simple background"
)

# ---- Optional real character-movement animation (AnimateDiff) ----------
# Off by default: on a 6GB card this reliably OOMs or is extremely slow.
# Enable with `--animation-mode ai` on a bigger GPU (e.g. AWS EC2 g4dn.xlarge
# [T4 16GB] or g5.xlarge [A10G 24GB]). Uses the SAME SD1.5 cartoon checkpoint
# (SD_MODEL_ID above) as the base, so the art style stays consistent with the
# still images - only a motion-adapter module is added on top.
ANIMATEDIFF_ADAPTER_ID = "guoyww/animatediff-motion-adapter-v1-5-2"
AI_VIDEO_NUM_FRAMES = 16
AI_VIDEO_FPS = 8
AI_VIDEO_GUIDANCE_SCALE = 7.5
AI_VIDEO_STEPS = 25

NEGATIVE_PROMPT = (
    "photorealistic, realistic, photo, 3d render, blurry, deformed, extra limbs, "
    "text, watermark, signature, writing, letters, words, low quality, ugly, "
    "multiple animals, group, crowd, many characters, duplicate, clone, "
    "pattern, seamless pattern, tile, tiled, wallpaper, texture, repeating, grid, "
    "human, person, people, man, woman, girl, boy, child"
)
