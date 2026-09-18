import json
from pathlib import Path

d = Path(r"C:\Users\tiajungba\.cline\data\workspaces\chat\manhwa-recap\test_recap\guided_out")
art = json.loads((d / "panels.json").read_text("utf-8"))
ps = art["panels"]
print(f"panels: {len(ps)}")
spoken = [p for p in ps if (p.get("narration") or "").strip()]
print(f"with narration: {len(spoken)}")
for p in ps:
    n = (p.get("narration") or "").strip()
    h = p["y_end"] - p["y_start"]
    flag = " [ctx]" if p.get("context_only") else (" [blank]" if p.get("blank_flag") == "blank" else "")
    nshow = n[:68] + ("..." if len(n) > 68 else "")
    print(f"  {p['id']}  h={h:6d}  {nshow!r}{flag}")
imgs = sorted(d.glob("panel_*.png"))
print(f"panel PNGs on disk: {len(imgs)}")
