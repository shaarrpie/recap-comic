"""Project the measured 10-page test timings to a full 26-page chapter.

All inputs are MEASURED from the test run; only the page count changes.
"""
PAGES_TEST = 10
PAGES_FULL = 26

# measured (seconds) from artifact mtimes, 10-page run
measured = {
    "Phase 1  AI pre-read (xkiro vision)": 491.9,
    "Phase 2  guided cut": 22.3,
    "Stage 1  build narration": 70.1,
    "Stage 2  TTS (28 edge-tts clips)": 57.7,
    "Stage 3  timeline + SRT": 0.5,
    "Stage 4  ffmpeg render": 747.3,
}
total_test = sum(measured.values())

# per-page normalizers measured on the 10-page run
per_page_ai = measured["Phase 1  AI pre-read (xkiro vision)"] / PAGES_TEST
per_page_cut = measured["Phase 2  guided cut"] / PAGES_TEST
panels_test = 28                     # timeline entries from 10 pages
panels_per_page = panels_test / PAGES_TEST
video_test = 565.6                   # seconds of output video
render_factor = 747.3 / video_test   # render wall per second of video

print(f"MEASURED (10 pages): {total_test:.0f}s = {total_test/60:.1f} min\n")

print(f"{'stage':38s} {'10pg (meas)':>12s} {'26pg (proj)':>12s}")
print("-" * 64)
proj = {}
for name, secs in measured.items():
    if "AI pre-read" in name:
        p = per_page_ai * PAGES_FULL
    elif "guided cut" in name:
        p = per_page_cut * PAGES_FULL
    elif "narration" in name or "TTS" in name:
        # both scale with panel count, not pages
        p = secs * (PAGES_FULL / PAGES_TEST)
    elif "render" in name:
        video_full = video_test * (PAGES_FULL / PAGES_TEST)
        p = video_full * render_factor
    else:
        p = secs
    proj[name] = p
    print(f"{name:38s} {secs:11.0f}s {p:11.0f}s")

total_full = sum(proj.values())
print("-" * 64)
print(f"{'TOTAL':38s} {total_test:11.0f}s {total_full:11.0f}s")
print(f"{'':38s} {total_test/60:10.1f}m {total_full/60:10.1f}m")
print(f"\nfull chapter = {total_full/60:.0f} minutes  ({total_full/3600:.2f} h)")

print("\n--- derived normalizers ---")
print(f"  AI pre-read : {per_page_ai:.1f}s/page (vision calls on 2000px chunks)")
print(f"  panels      : {panels_per_page:.1f}/page -> {panels_per_page*PAGES_FULL:.0f} panels in a chapter")
video_full = video_test * (PAGES_FULL / PAGES_TEST)
print(f"  video length: {video_full/60:.0f} min of output")
print(f"  render      : {render_factor:.2f}x realtime (blur+vignette, chunked, veryfast)")

print("\n--- where the time goes (26-page chapter) ---")
for name, p in sorted(proj.items(), key=lambda kv: -kv[1]):
    print(f"  {p/total_full*100:4.0f}%  {name}")
