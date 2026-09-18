import json
import subprocess
from pathlib import Path

import numpy as np

work = Path(r"C:\Users\tiajungba\.cline\data\workspaces\chat\manhwa-recap\test_recap")
exe = r"C:\Users\tiajungba\AppData\Local\Programs\Python\Python311\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"

mp4 = work / "recap.mp4"
print("=== OUTPUT ===")
print(mp4, mp4.stat().st_size / 1e6, "MB")

p = subprocess.run([exe, "-i", str(mp4)], capture_output=True, text=True)
for line in p.stderr.splitlines():
    if "Duration" in line or "Stream #" in line:
        print(" ", line.strip())

# --- check suspicious TTS durations ---
narr = json.loads((work / "narration.json").read_text("utf-8"))
audio = json.loads((work / "audio" / "audio.json").read_text("utf-8"))
by_text = {e["id"]: e["text"] for e in narr["entries"]}
print("\n=== longest TTS clips (text vs measured duration) ===")
rows = sorted(audio["entries"], key=lambda e: -e["duration_seconds"])[:6]
for e in rows:
    t = by_text.get(e["entry_id"], "")
    wpm = len(t.split()) / e["duration_seconds"] * 60 if e["duration_seconds"] else 0
    print(f"  {e['entry_id']:8s} {e['duration_seconds']:7.2f}s  words={len(t.split()):4d}  wpm={wpm:5.0f}")
    print(f"           text: {t[:110]!r}")

# --- frame check: vignette + blur applied? ---
def frame(path, t):
    pr = subprocess.run(
        [exe, "-y", "-ss", str(t), "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True)
    return np.frombuffer(pr.stdout, dtype=np.uint8).reshape(1920, 1080, 3)

print("\n=== style spot-check (t=5s and t=120s) ===")
for t in (5.0, 120.0, 400.0):
    try:
        f = frame(mp4, t)
    except ValueError:
        print(f"  t={t}: no frame")
        continue
    center = f[800:1120, 400:680].reshape(-1, 3).mean(0)
    edge = np.concatenate([f[:60].reshape(-1, 3), f[-60:].reshape(-1, 3),
                           f[:, :40].reshape(-1, 3), f[:, -40:].reshape(-1, 3)]).mean(0)
    print(f"  t={t:5.0f}s  center={center.round(0)}  edge={edge.round(0)}  "
          f"edge_darker={bool((edge < center - 8).any())}")

# --- SRT sanity ---
srt = (work / "recap.srt").read_text("utf-8")
cues = [l for l in srt.splitlines() if "-->" in l]
print(f"\n=== captions: {len(cues)} cues; first/last ===")
print("  first:", cues[0])
print("  last :", cues[-1])
