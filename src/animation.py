"""Simple camera-movement animation (zoom/pan) applied to a still image via ffmpeg.

IMPORTANT ffmpeg quirk discovered during testing (this exact build: ffmpeg 7.1
"essentials" from gyan.dev): the `zoompan` filter's `x`/`y` parameters do not
work at all - verified with 6+ isolated tests (constant values, on-based
formulas, self-referencing formulas, combined with a changing zoom) and every
single one produced pixel-identical output frames. Only `zoompan`'s `zoom`
parameter actually animates. Because of this:
  - zoom_in / zoom_out use `zoompan` (zoom only, no x/y) - this does work, but
    anchors at the top-left corner rather than centering, since x/y can't be
    set. Canvas excess is kept modest to limit how far that drift goes.
  - pan_left / pan_right / zoom_pan use the `crop` filter instead, with
    time-varying `x`/`y` expressions (crop's x/y DO support per-frame `t`,
    confirmed with a 96%-of-pixels-changed diff test) over a statically
    pre-scaled, larger-than-output canvas. `crop`'s width/height can't be
    animated (ffmpeg only evaluates them once at init, no `eval=frame` option
    exists for `crop` unlike `scale`), which is why this path is position-only
    (pure pan), not zoom.
"""
import shutil
import subprocess

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

MOTIONS = ["zoom_in", "pan_left", "zoom_out", "pan_right", "zoom_in", "zoom_pan"]

ZOOM_MOTIONS = {"zoom_in", "zoom_out"}


def motion_for_scene(index: int) -> str:
    """index is 0-based scene index."""
    return MOTIONS[index % len(MOTIONS)]


def _zoom_filter(motion: str, frames: int, size: int) -> str:
    canvas = int(size * 1.3)
    d = frames
    if motion == "zoom_in":
        zoom_expr = f"'1.0+0.18*on/{d}'"
    else:  # zoom_out
        zoom_expr = f"'1.18-0.18*on/{d}'"
    return (
        f"scale={canvas}:{canvas}:flags=lanczos,"
        f"zoompan=z={zoom_expr}:d={d}:s={size}x{size}:fps=25,"
        f"unsharp=5:5:0.6:5:5:0.0"
    )


def _pan_filter(motion: str, duration: float, size: int) -> str:
    canvas = int(size * 1.3)
    margin = canvas - size
    d = max(duration, 0.001)
    if motion == "pan_left":
        x_expr = f"'{margin}*(1-t/{d})'"
        y_expr = f"'{margin}/2'"
    elif motion == "pan_right":
        x_expr = f"'{margin}*(t/{d})'"
        y_expr = f"'{margin}/2'"
    elif motion == "zoom_pan":
        # Diagonal pan (crop's w/h can't be animated, see module docstring,
        # so true zoom+pan isn't available here - diagonal drift stands in
        # for the "slight zoom + transition" scene type instead).
        x_expr = f"'{margin}*(t/{d})'"
        y_expr = f"'{margin}*(t/{d})'"
    else:
        raise ValueError(f"unknown pan motion: {motion}")
    return (
        f"scale={canvas}:{canvas}:flags=lanczos,"
        f"crop={size}:{size}:x={x_expr}:y={y_expr},"
        f"unsharp=5:5:0.6:5:5:0.0"
    )


def animate_scene(image_path, out_path, duration: float, motion: str, fps: int = 25, size: int = 768):
    """Create a short mp4 clip from a still image with subtle zoom/pan + fade transitions."""
    frames = max(1, round(duration * fps))

    if motion in ZOOM_MOTIONS:
        motion_filter = _zoom_filter(motion, frames, size)
    else:
        motion_filter = _pan_filter(motion, duration, size)

    fade_dur = min(0.35, duration / 4)
    fade_out_start = max(0, duration - fade_dur)
    vf = (
        f"{motion_filter},"
        f"fade=t=in:st=0:d={fade_dur},fade=t=out:st={fade_out_start}:d={fade_dur}"
    )
    cmd = [
        FFMPEG, "-y", "-loop", "1", "-i", str(image_path),
        "-vf", vf, "-t", f"{duration:.3f}",
        "-r", str(fps), "-pix_fmt", "yuv420p", "-an",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg animation failed for {image_path}:\n{result.stderr[-2000:]}")
    return out_path
