"""Combine animated scene clips + Hindi narration + subtitles into the final MP4 via ffmpeg."""
import shutil
import subprocess
from pathlib import Path

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

SUBTITLE_FONT = "Nirmala UI"  # ships with Windows, supports Devanagari (Hindi)


def _run(cmd, step_name):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg step '{step_name}' failed:\n{result.stderr[-3000:]}")
    return result


def concat_videos(video_paths, out_path, work_dir):
    list_file = Path(work_dir) / "concat_list.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for p in video_paths:
            abs_path = str(Path(p).resolve()).replace("\\", "/")
            f.write(f"file '{abs_path}'\n")
    cmd = [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
           "-c", "copy", str(out_path)]
    _run(cmd, "concat_videos")
    return out_path


def build_audio_track(scene_audio_paths, clip_durations, out_path):
    """Pad each scene's narration to its clip duration with trailing silence, concatenate."""
    from pydub import AudioSegment

    combined = AudioSegment.silent(duration=0)
    for audio_path, clip_dur in zip(scene_audio_paths, clip_durations):
        seg = AudioSegment.from_wav(str(audio_path))
        target_ms = int(round(clip_dur * 1000))
        pad_ms = max(0, target_ms - len(seg))
        seg = seg + AudioSegment.silent(duration=pad_ms)
        combined += seg
    combined.export(str(out_path), format="wav")
    return out_path


def mux_audio_video(video_path, audio_path, out_path):
    cmd = [
        FFMPEG, "-y", "-i", str(video_path), "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(out_path),
    ]
    _run(cmd, "mux_audio_video")
    return out_path


def _escape_subtitle_path(path) -> str:
    p = str(Path(path).resolve()).replace("\\", "/")
    p = p.replace(":", "\\:")
    return p


def burn_subtitles(video_path, srt_path, out_path, font=SUBTITLE_FONT):
    srt_escaped = _escape_subtitle_path(srt_path)
    style = (
        f"FontName={font},FontSize=20,PrimaryColour=&H00FFFFFF&,"
        f"OutlineColour=&H00000000&,BorderStyle=3,Outline=1,Shadow=0,MarginV=40"
    )
    vf = f"subtitles='{srt_escaped}':force_style='{style}'"
    cmd = [
        FFMPEG, "-y", "-i", str(video_path),
        "-vf", vf, "-c:a", "copy", str(out_path),
    ]
    _run(cmd, "burn_subtitles")
    return out_path
