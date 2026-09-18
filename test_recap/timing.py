import json
import os
from pathlib import Path
from datetime import datetime

work = Path(r"C:\Users\tiajungba\.cline\data\workspaces\chat\manhwa-recap\test_recap")


def mtime(p):
    return datetime.fromtimestamp(p.stat().st_mtime)


print("=== artifact mtimes (wall-clock stage boundaries) ===")
artifacts = [
    ("strip.png (stitched)", work / "strip.png"),
    ("plan.json (Phase 1 AI pre-read)", work / "guided_out" / "plan.json"),
    ("panels.json (Phase 2 cut)", work / "guided_out" / "panels.json"),
    ("narration.json", work / "narration.json"),
    ("audio/audio.json", work / "audio" / "audio.json"),
    ("timeline.json", work / "timeline.json"),
    ("recap.srt", work / "recap.srt"),
    ("recap.mp4 (render done)", work / "recap.mp4"),
]
times = []
for label, p in artifacts:
    if p.is_file():
        t = mtime(p)
        times.append((label, t))
        print(f"  {t:%H:%M:%S}  {label}")

print("\n=== measured stage durations ===")
for (l1, t1), (l2, t2) in zip(times, times[1:]):
    d = (t2 - t1).total_seconds()
    print(f"  {d:7.1f}s  {l1.split(' (')[0]:28s} -> {l2.split(' (')[0]}")

total = (times[-1][1] - times[0][1]).total_seconds()
print(f"  {total:7.1f}s  TOTAL (10 pages)")

# scale factors
tl = json.loads((work / "timeline.json").read_text("utf-8"))
n_panels = len(tl["entries"])
dur = sum(e["duration_seconds"] for e in tl["entries"])
print(f"\n=== normalizers ===")
print(f"  pages=10  panels={n_panels}  video_duration={dur:.1f}s")
print(f"  per page : {total/10:.1f}s wall")
print(f"  per panel: {total/n_panels:.1f}s wall")
print(f"  render realtime factor: {12.5*60/dur:.2f}x (render wall / video length)")

# clip count / sizes
audio = json.loads((work / "audio" / "audio.json").read_text("utf-8"))
clips = list((work / "audio").glob("*.mp3"))
print(f"  tts clips={len(clips)}  avg clip={dur/len(clips):.1f}s")
