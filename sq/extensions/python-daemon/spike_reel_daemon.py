#!/usr/bin/env python3
"""
SYNAPSE-QUANTUM Node 15: autonomous spike-burst reel engine.

  python spike_reel_daemon.py --watch ~/Downloads            # daemon: process every new *.webm once
  python spike_reel_daemon.py raw.webm                       # one-shot

Fixes: it really watches a directory now (the old docstring claimed it did), waits until a file has
stopped growing, never re-processes a file, scales to 1080x1920 after the 9:16 crop, strips audio
(canvas recordings have none, so the "audio RMS" idea in the blueprint does not apply), and only
burns in a caption if a usable font exists. If a sidecar "<name>.events.json" exists
({"bursts":[{"t":12.3,"dur":6.0}, ...]}) each burst becomes its own reel; otherwise the whole clip
is used.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "C:/Windows/Fonts/consola.ttf",
]


def find_font():
    return next((f for f in FONT_CANDIDATES if Path(f).exists()), None)


def build_filter(caption: str) -> str:
    vf = "crop='min(iw,ih*9/16)':ih,scale=1080:1920:flags=lanczos"
    font = find_font()
    if font:
        safe = caption.replace(":", "\\:").replace("'", "")
        vf += f",drawtext=fontfile='{font}':text='{safe}':fontcolor=cyan:fontsize=40:x=40:y=80"
    return vf


def run_ffmpeg(src: Path, out: Path, start: float = 0.0, dur: float | None = None) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if dur:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-vf", build_filter("SYNAPSE-QUANTUM SPIKE BURST"), "-an",
            "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, check=True)
    print(f"[Node 15] exported {out}")


def process(src: Path) -> None:
    events = src.with_suffix(".events.json")
    if events.exists():
        bursts = json.loads(events.read_text()).get("bursts", [])
        for i, b in enumerate(bursts):
            run_ffmpeg(src, src.with_name(f"{src.stem}_burst{i + 1}_9x16.mp4"), max(0.0, b["t"] - 1.0), b.get("dur", 6.0) + 2.0)
        if bursts:
            return
    run_ffmpeg(src, src.with_name(f"{src.stem}_9x16.mp4"))


def wait_until_stable(path: Path, checks: int = 3, interval: float = 1.0) -> bool:
    last, stable = -1, 0
    while stable < checks:
        if not path.exists():
            return False
        size = path.stat().st_size
        stable = stable + 1 if size == last and size > 0 else 0
        last = size
        time.sleep(interval)
    return True


def watch(directory: Path) -> None:
    seen = {p for p in directory.glob("*.webm")}   # ignore pre-existing files
    print(f"[Node 15] watching {directory} for new .webm recordings (Ctrl+C to stop)")
    while True:
        for p in sorted(directory.glob("*.webm")):
            if p in seen:
                continue
            seen.add(p)
            if wait_until_stable(p):
                try:
                    process(p)
                except subprocess.CalledProcessError as exc:
                    print(f"[Node 15] ffmpeg failed for {p.name}: {exc}", file=sys.stderr)
        time.sleep(2.0)


def main() -> None:
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH")
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", help="single .webm to process")
    ap.add_argument("--watch", type=Path, help="directory to watch")
    a = ap.parse_args()
    if a.watch:
        watch(a.watch)
    elif a.input:
        src = Path(a.input)
        if not src.exists():
            sys.exit(f"not found: {src}")
        process(src)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
