"""Augment the measured style_profile.json with the interpretation and the
explicit pipeline mapping (measured value -> required config change).

Run after analyze_style.py. Writes style_profile_final.json.
"""
import json

RAW = json.load(open("style_ref/style_profile.json"))

interp = {
    "verdict": (
        "Clean, restrained landscape edit. Almost every default effect in "
        "this pipeline's 'dynamic' style is ABSENT from the reference. "
        "Matching it means subtracting effects plus switching to 16:9."
    ),

    # What the measurements mean, and how confident each reading is.
    "findings": {
        "aspect": {
            "measured": "16:9 landscape, 1292x706, no letterbox/pillarbox (bars=0 on all edges)",
            "confidence": "high (read from container + verified zero black bars)",
            "implication": "Biggest single change. All renderers here are hardcoded 9:16 portrait.",
        },
        "color": {
            "measured": "saturation 0.109 (muted), mean RGB ~(140.7,137.8,137.8) neutral, no color cast, brightness 60.2-233.9 wide range",
            "confidence": "high (per-pixel stats over all 26 sampled frames)",
            "implication": "Neutral white balance, desaturated grade. Pipeline default boosts saturation 1.05 and applies teal-shadow/orange-highlight split -- both must go.",
        },
        "transitions": {
            "measured": "10 hard cuts, avg shot 2.33s; diff distribution has NO dissolve plateau between 'motion' and 'cut' bands",
            "confidence": "medium-high (scene detection + diff histogram; a slow dissolve under 25 diff could hide in the motion band)",
            "implication": "Use plain cuts. Pipeline default inserts a white-flash glitch between every panel -- remove it.",
        },
        "pacing": {
            "measured": "shot lengths 0.6, 2.4, 2.6, 4.0, 4.4, 2.8, 1.8, 2.0, 2.0, 1.8, 1.2 s",
            "confidence": "high (derived from cut times)",
            "implication": "Fast hook, long mid holds, accelerating outro. min_display 2.0s is too high for the 0.6s intro hook; lower to 1.8s and drive durations from content.",
        },
        "motion": {
            "measured": "0 static samples out of 127; Ken Burns zoom 1.10x in 3 bursts (t=0.0-0.4, 17.0-18.4, 20.8-22.4); translational drift +-1.8px, direction flips per segment",
            "confidence": "high (phase correlation + crop-scale correlation)",
            "implication": "Always moving, but gently. Zoom 1.10x, NOT the default punch-zoom 1.20 with settle. The drifting is jitter, not a deliberate scroll -- do not add a steady pan.",
        },
        "captions": {
            "measured": "white text on near-black (fg ~[254,254,254], bg ~[7,7,7]), horizontally centered, thin band ~5% frame height, found in only 4/26 frames (t~6-10s, upper area)",
            "confidence": "LOW -- the detector only matches dark-band + bright-text; stylized or non-boxed captions would be missed. Cannot confirm semantics (title vs subtitle) because this model cannot see frames.",
            "implication": "If captions are burned in: white, centered, no background box, thin. Likely only over some segments. SRT sidecar path is unaffected.",
        },
    },

    # Explicit mapping table: measured -> required change.
    "pipeline_mapping": [
        {
            "dimension": "aspect ratio",
            "measured": "16:9 (1292x706)",
            "knob": "adapters/render_ffmpeg.py WIDTH,HEIGHT ; CinematicConfig.output_width/height",
            "current": "1080x1920 portrait (hardcoded)",
            "required": "1920x1080 landscape",
            "risk": "fit_pan() overflow axis flips; geometry tests must be re-verified",
        },
        {
            "dimension": "saturation",
            "measured": 0.109,
            "knob": "CinematicConfig.grade_saturation",
            "current": 1.05,
            "required": 0.88,
            "risk": "none",
        },
        {
            "dimension": "white balance",
            "measured": "neutral, no cast",
            "knob": "grade_shadows / grade_highlights",
            "current": "-0.04 / +0.02 teal-orange split",
            "required": "0.0 / 0.0",
            "risk": "none",
        },
        {
            "dimension": "contrast",
            "measured": "natural in-camera",
            "knob": "grade_contrast",
            "current": 1.08,
            "required": 1.00,
            "risk": "none",
        },
        {
            "dimension": "transitions",
            "measured": "10 hard cuts, no dissolves",
            "knob": "CinematicConfig.glitch_enabled ; render_ffmpeg transitions",
            "current": "white-flash glitch between every panel / xfade available",
            "required": "glitch_enabled=False, transitions=[{'type':'cut'}]",
            "risk": "none",
        },
        {
            "dimension": "zoom",
            "measured": "1.10x Ken Burns in bursts",
            "knob": "kb_zoom_end / punch_scale / punch_frames",
            "current": "punch 1.20 -> settle 1.06 -> creep 1.12",
            "required": "kb_zoom_end=1.10, punch_frames=0",
            "risk": "none",
        },
        {
            "dimension": "screen shake",
            "measured": "absent (drift <=1.8px)",
            "knob": "shake_enabled",
            "current": "True (amplitude 12px)",
            "required": "False",
            "risk": "none",
        },
        {
            "dimension": "vignette / letterbox / speed lines",
            "measured": "all absent",
            "knob": "vignette_enabled / letterbox_enabled / speedlines_enabled",
            "current": "True / False / True",
            "required": "False / False / False",
            "risk": "none",
        },
        {
            "dimension": "pacing",
            "measured": "avg 2.33s/shot, 0.6s hook, 1.8s outro",
            "knob": "adapters/timeline.py build() min_display, gap",
            "current": "min_display=2.0, gap=0.35",
            "required": "min_display=1.8, gap=0.0",
            "risk": "short panels may under-display narration; keep audio-driven durations",
        },
    ],

    "caveats": [
        "This model cannot ingest images (Read tool rejected frames as "
        "unsupported input). Every finding above is statistical, not semantic "
        "-- content, composition quality, and on-screen meaning were never "
        "verified by eye.",
        "Caption detection matched only 4/26 frames and uses a narrow "
        "dark-band+bright-text signature; real captions may be missed.",
        "scene detection threshold 0.25; slow dissolves could be classified "
        "as motion. The diff histogram showed no dissolve plateau, which is "
        "evidence for hard cuts but not proof.",
        "Audio envelope was extracted but speech/narration style was NOT "
        "transcribed or analyzed for tone -- a TTS voice/style match needs a "
        "transcription pass first.",
    ],
}

out = dict(RAW)
out["interpretation"] = interp
with open("style_ref/style_profile_final.json", "w") as fh:
    json.dump(out, fh, indent=2)

print("wrote style_ref/style_profile_final.json")
print("mapping rows:", len(interp["pipeline_mapping"]))
for row in interp["pipeline_mapping"]:
    print(f"  - {row['dimension']:22} {row['current']!r:38} -> {row['required']!r}")
