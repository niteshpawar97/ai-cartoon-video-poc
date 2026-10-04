"""Environment / hardware check for the AI Cartoon Video PoC.

Run with the project's venv:
    .venv/Scripts/python check_environment.py
"""
import platform
import shutil
import subprocess
import sys


def human_gb(num_bytes: float) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def get_ram():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return human_gb(vm.total), human_gb(vm.available)
    except ImportError:
        return "psutil not installed", "?"


def get_disk(path="."):
    total, used, free = shutil.disk_usage(path)
    return human_gb(total), human_gb(free)


def get_ffmpeg():
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=10)
        first_line = out.stdout.splitlines()[0] if out.stdout else "unknown"
        return exe, first_line
    except Exception as e:
        return exe, f"error: {e}"


def get_gpu():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        line = out.stdout.strip().splitlines()[0]
        name, mem_total, mem_used, mem_free, driver = [p.strip() for p in line.split(",")]
        return {
            "name": name, "mem_total": mem_total, "mem_used": mem_used,
            "mem_free": mem_free, "driver": driver,
        }
    except Exception:
        return None


def get_torch_cuda():
    try:
        import torch
        info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["device_name"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            info["vram_free"] = human_gb(free)
            info["vram_total"] = human_gb(total)
        return info
    except ImportError:
        return None


def main():
    print("=============================")
    print("SYSTEM")
    print("=============================\n")

    print(f"OS:          {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Python:      {sys.version.split()[0]} ({sys.executable})")

    ram_total, ram_avail = get_ram()
    print(f"RAM:         total={ram_total}, available={ram_avail}")

    gpu = get_gpu()
    if gpu:
        print(f"\nGPU:         {gpu['name']}")
        print(f"VRAM:        total={gpu['mem_total']}, used={gpu['mem_used']}, free={gpu['mem_free']}")
        print(f"Driver:      {gpu['driver']}")
    else:
        print("\nGPU:         not detected (nvidia-smi not found)")
        print("VRAM:        N/A")

    torch_info = get_torch_cuda()
    if torch_info is None:
        print("\nPyTorch:     not installed")
        print("CUDA:        N/A")
    else:
        print(f"\nPyTorch:     {torch_info['torch_version']}")
        print(f"CUDA avail:  {torch_info['cuda_available']}")
        if torch_info["cuda_available"]:
            print(f"CUDA ver:    {torch_info['cuda_version']}")
            print(f"CUDA device: {torch_info['device_name']}")
            print(f"CUDA VRAM:   free={torch_info['vram_free']} / total={torch_info['vram_total']}")

    ff = get_ffmpeg()
    if ff:
        print(f"\nFFmpeg:      {ff[0]}")
        print(f"             {ff[1]}")
    else:
        print("\nFFmpeg:      NOT FOUND")

    disk_total, disk_free = get_disk(".")
    print(f"\nDisk (project drive): total={disk_total}, free={disk_free}")

    print("\n=============================")

    # Practical recommendation block
    print("\nRECOMMENDATION")
    print("-----------------------------")
    if gpu:
        free_mb = float(gpu["mem_free"].split()[0])
        if free_mb < 3500:
            print("WARNING: less than ~3.5GB free VRAM right now.")
            print("Close other GPU apps (browser hardware accel, games) before running image generation.")
        print("GPU detected with <= 8GB VRAM class card -> recommended stack:")
        print(" - Image gen: Stable Diffusion 1.5 (fp16, attention slicing), 512x512")
        print(" - LLM: small GGUF (<=3B) via llama-cpp-python, CPU inference (fast on Ryzen/Intel 8c+)")
        print(" - TTS: Hindi VITS (facebook/mms-tts-hin) - lightweight, CPU or GPU")
    else:
        print("No NVIDIA GPU detected -> all inference will fall back to CPU (slower).")


if __name__ == "__main__":
    main()
