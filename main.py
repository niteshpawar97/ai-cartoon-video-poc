"""AI Cartoon Video PoC - CLI entrypoint.

Usage:
    python main.py --story input/story.txt
    python main.py --story input/story.txt --duration 30 --scenes 6
"""
import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from src import config

STAGES = [
    "story_processing", "scene_generation", "image_generation",
    "tts_generation", "animation", "subtitle_generation", "ffmpeg", "total",
]


def setup_run_dirs(run_id: str) -> Path:
    run_dir = config.OUTPUT_DIR / f"run-{run_id}"
    for d in ["input", "scenes", "images", "audio", "animation", "subtitles", "final", "logs"]:
        (run_dir / d).mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("cartoon_poc")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def get_gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            _, total = torch.cuda.mem_get_info(0)
            return name, f"{total / 1024 ** 3:.1f} GB"
    except Exception:
        pass
    return "N/A", "N/A"


def main():
    parser = argparse.ArgumentParser(description="Local AI cartoon video generator PoC")
    parser.add_argument("--story", default=str(config.INPUT_DIR / "story.txt"))
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--scenes", type=int, default=6)
    parser.add_argument(
        "--animation-mode", choices=["camera", "ai"], default="camera",
        help=(
            "'camera' (default): ffmpeg zoom/pan over still images, works on "
            "any GPU including 6GB cards. 'ai': real character/object motion "
            "via AnimateDiff - needs a much bigger GPU (e.g. AWS EC2 "
            "g4dn.xlarge/g5.xlarge), not practical on 6GB."
        ),
    )
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = setup_run_dirs(run_id)
    logger = setup_logging(run_dir / "logs" / "process.log")

    timings = {k: 0.0 for k in STAGES}
    status = "failed"
    error_msg = None
    scenes_data = None
    clip_durations = []
    final_path = None
    started_at = datetime.now().isoformat()
    t_total_start = time.time()

    try:
        t0 = time.time()
        story_path = Path(args.story)
        story_text = story_path.read_text(encoding="utf-8").strip()
        shutil.copy(story_path, run_dir / "input" / "story.txt")
        logger.info(f"Run {run_id} started. Story: {len(story_text)} chars from {story_path}")
        timings["story_processing"] = time.time() - t0

        # ---- scene generation (local LLM) ----
        t0 = time.time()
        from src import story as story_mod
        scenes_data = story_mod.generate_scenes(
            story_text, num_scenes=args.scenes, target_duration=args.duration
        )
        story_mod.unload_llm()
        timings["scene_generation"] = time.time() - t0
        logger.info(
            f"Scene generation done in {timings['scene_generation']:.2f}s, "
            f"{len(scenes_data['scenes'])} scenes, title='{scenes_data['title']}'"
        )

        (run_dir / "scenes" / "scenes.json").write_text(
            json.dumps(scenes_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_dir / "scenes" / "characters.json").write_text(
            json.dumps({"character_description": scenes_data["character_description"]},
                       ensure_ascii=False, indent=2), encoding="utf-8"
        )
        visual_prompts = [
            {"scene": s["scene"], "visual_prompt": s["visual_prompt"],
             "visual_prompt_full": s["visual_prompt_full"]}
            for s in scenes_data["scenes"]
        ]
        (run_dir / "scenes" / "visual_prompts.json").write_text(
            json.dumps(visual_prompts, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # ---- image generation (local SD) ----
        t0 = time.time()
        from src import image_generation as img_mod
        image_paths = []
        for s in scenes_data["scenes"]:
            idx = s["scene"]
            out_path = run_dir / "images" / f"scene_{idx:03d}.png"
            ti = time.time()
            img_mod.generate_image(s["visual_prompt_full"], out_path, seed=42)
            image_paths.append(out_path)
            logger.info(f"Image scene {idx} generated in {time.time()-ti:.2f}s -> {out_path.name}")
        img_mod.unload_pipeline()
        timings["image_generation"] = time.time() - t0

        # ---- Hindi TTS (local VITS) ----
        t0 = time.time()
        from src import tts as tts_mod
        audio_entries = []
        for s in scenes_data["scenes"]:
            idx = s["scene"]
            out_path = run_dir / "audio" / f"scene_{idx:03d}.wav"
            dur = tts_mod.generate_audio(s["narration_hindi"], out_path)
            s["tts_duration"] = dur
            audio_entries.append({
                "scene": idx, "text": s["narration_hindi"],
                "duration": dur, "file": out_path.name,
            })
            logger.info(f"TTS scene {idx}: {dur:.2f}s audio")
        tts_mod.unload_tts()
        timings["tts_generation"] = time.time() - t0
        (run_dir / "audio" / "narration.json").write_text(
            json.dumps(audio_entries, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # ---- animation: ffmpeg camera zoom/pan, or real AI motion ----
        t0 = time.time()
        n_scenes = len(scenes_data["scenes"])
        total_narration = sum(s["tts_duration"] for s in scenes_data["scenes"])
        # Spread any shortfall between total narration and the requested video
        # length across scenes as breathing-room padding, so the final video
        # lands close to --duration instead of just the (often shorter) sum of
        # narration clips. Clamped so very short stories don't get oddly long
        # silences, and very long narration still just pads minimally.
        MIN_PAD, MAX_PAD = 0.3, 2.5
        shortfall = max(0.0, args.duration - total_narration)
        pad = min(MAX_PAD, max(MIN_PAD, shortfall / n_scenes))
        anim_paths = []

        if args.animation_mode == "ai":
            from src import ai_animation as ai_anim_mod
            for i, s in enumerate(scenes_data["scenes"]):
                idx = s["scene"]
                clip_dur = s["tts_duration"] + pad
                clip_durations.append(clip_dur)
                raw_path = run_dir / "animation" / f"_raw_scene_{idx:03d}.mp4"
                out_path = run_dir / "animation" / f"scene_{idx:03d}.mp4"
                motion_prompt = f"{s['visual_prompt_full']}, smooth motion"
                ai_anim_mod.generate_scene_clip(motion_prompt, raw_path, seed=42)
                ai_anim_mod.normalize_clip(raw_path, out_path, clip_dur)
                anim_paths.append(out_path)
                logger.info(f"AI-animated scene {idx} ({clip_dur:.2f}s)")
            ai_anim_mod.unload_pipeline()
        else:
            from src import animation as anim_mod
            for i, s in enumerate(scenes_data["scenes"]):
                idx = s["scene"]
                clip_dur = s["tts_duration"] + pad
                clip_durations.append(clip_dur)
                motion = anim_mod.motion_for_scene(i)
                out_path = run_dir / "animation" / f"scene_{idx:03d}.mp4"
                anim_mod.animate_scene(image_paths[i], out_path, clip_dur, motion)
                anim_paths.append(out_path)
                logger.info(f"Animated scene {idx} ({motion}, {clip_dur:.2f}s)")
        timings["animation"] = time.time() - t0

        # ---- subtitles ----
        t0 = time.time()
        from src import subtitles as sub_mod
        entries = []
        cursor = 0.0
        for s, clip_dur in zip(scenes_data["scenes"], clip_durations):
            entries.append({
                "text": s["narration_hindi"],
                "start": cursor, "end": cursor + s["tts_duration"],
            })
            cursor += clip_dur
        srt_path = run_dir / "subtitles" / "subtitles.srt"
        sub_mod.build_srt(entries, srt_path)
        timings["subtitle_generation"] = time.time() - t0
        logger.info(f"Subtitles written -> {srt_path}")

        # ---- ffmpeg: concat + mux audio + burn subtitles ----
        t0 = time.time()
        from src import video as video_mod
        concat_path = run_dir / "final" / "_concat_silent.mp4"
        video_mod.concat_videos(anim_paths, concat_path, run_dir / "final")

        audio_track_path = run_dir / "final" / "_full_audio.wav"
        scene_audio_paths = [run_dir / "audio" / f"scene_{s['scene']:03d}.wav" for s in scenes_data["scenes"]]
        video_mod.build_audio_track(scene_audio_paths, clip_durations, audio_track_path)

        with_audio_path = run_dir / "final" / "_with_audio.mp4"
        video_mod.mux_audio_video(concat_path, audio_track_path, with_audio_path)

        final_path = run_dir / "final" / "final_video.mp4"
        video_mod.burn_subtitles(with_audio_path, srt_path, final_path)
        timings["ffmpeg"] = time.time() - t0
        logger.info(f"Final video -> {final_path}")

        status = "success"
    except Exception as e:
        logger.exception("Pipeline failed")
        error_msg = str(e)
    finally:
        timings["total"] = time.time() - t_total_start

    gpu_name, vram = get_gpu_info()
    video_duration = sum(clip_durations) if clip_durations else 0.0

    report = {
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(),
        "input_story": str(args.story),
        "animation_mode": args.animation_mode,
        "scene_count": len(scenes_data["scenes"]) if scenes_data else 0,
        "video_duration_sec": round(video_duration, 2),
        "gpu": gpu_name,
        "vram": vram,
        "timings": {k: round(v, 2) for k, v in timings.items()},
        "output_video": str(final_path) if final_path and final_path.exists() else None,
        "status": status,
    }
    if error_msg:
        report["error"] = error_msg
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if status == "success":
        latest_dir = config.OUTPUT_DIR / "latest"
        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        shutil.copytree(run_dir, latest_dir)
        logger.info(f"Copied outputs to {latest_dir}")

    print("\n========================================")
    print("RUN COMPLETE")
    print("========================================")
    print(f"Run ID:    {run_id}")
    print(f"Status:    {status}")
    print(f"\nGPU:       {gpu_name}")
    print(f"VRAM:      {vram}")
    print(f"\nScenes:         {report['scene_count']}")
    print(f"Video Duration: {report['video_duration_sec']:.1f} sec")
    print(f"\nStory processing:   {timings['story_processing']:.2f} sec")
    print(f"Scene Generation:   {timings['scene_generation']:.2f} sec")
    print(f"Image Generation:   {timings['image_generation']:.2f} sec")
    print(f"Hindi TTS:          {timings['tts_generation']:.2f} sec")
    print(f"Animation:          {timings['animation']:.2f} sec")
    print(f"Subtitles:          {timings['subtitle_generation']:.2f} sec")
    print(f"FFmpeg:             {timings['ffmpeg']:.2f} sec")
    print(f"\nTOTAL:              {timings['total']:.2f} sec")
    print(f"\nOutput: {report['output_video']}")
    print("========================================\n")

    sys.exit(0 if status == "success" else 1)


if __name__ == "__main__":
    main()
