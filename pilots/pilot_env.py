# pilots/pilot_env.py
"""Environment/version probe. Always runnable, no network, no binaries.
Run:  python pilots/pilot_env.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from importlib.metadata import version

PKGS = ["opencv-python-headless", "numpy", "pillow", "edge-tts", "moviepy",
        "imageio-ffmpeg", "pytesseract", "pydantic", "typer", "google-genai",
        "kokoro-onnx", "easyocr", "manga-ocr", "pytest"]


def main() -> None:
    print("python:", sys.version.split()[0])
    for p in PKGS:
        try:
            print(f"{p}=={version(p)}")
        except Exception:  # noqa: BLE001 - probe must not abort on one miss
            print(f"{p}==NOT INSTALLED")
    for binary in ("ffmpeg", "ffprobe", "tesseract"):
        print(binary, "->", shutil.which(binary) or "MISSING")
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        out = subprocess.run([exe, "-version"], capture_output=True,
                             text=True, check=False).stdout.splitlines()[0]
        print("imageio_ffmpeg bundled ffmpeg:", exe)
        print("  ", out)
    except Exception as exc:  # noqa: BLE001 - probe reports, never aborts
        print("imageio_ffmpeg unavailable:", exc)


if __name__ == "__main__":
    main()
