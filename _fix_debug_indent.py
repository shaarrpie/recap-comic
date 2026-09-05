from pathlib import Path

p = Path("debug_view.py")
s = p.read_text(encoding="utf-8")

old = """        # final snapped boundaries (green) — omitted in Phase 1 (no cuts yet)
        if cuts is not None:
            for c in cuts:
            for y in (max(0, c.y_start), min(h, c.y_end)):
                draw.line([(0, y), (w, y)], fill=(0, 190, 0), width=4)
            draw.text((w - 200, max(0, c.y_start + 4)), c.id,
                      fill=(0, 140, 0), font=small)"""

new = """        # final snapped boundaries (green) — omitted in Phase 1 (no cuts yet)
        if cuts is not None:
            for c in cuts:
                for y in (max(0, c.y_start), min(h, c.y_end)):
                    draw.line([(0, y), (w, y)], fill=(0, 190, 0), width=4)
                draw.text((w - 200, max(0, c.y_start + 4)), c.id,
                          fill=(0, 140, 0), font=small)"""

assert old in s, "old indent block not found"
s = s.replace(old, new)
p.write_text(s, encoding="utf-8")
print("OK")
