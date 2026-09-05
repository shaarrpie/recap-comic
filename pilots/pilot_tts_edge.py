# pilots/pilot_tts_edge.py
"""Network pilot for the edge-tts adapter (edge-tts==7.2.8).
Synthesizes one short line, saves mp3, prints WordBoundary events.
Run:  python pilots/pilot_tts_edge.py   (requires internet access to
Microsoft's TTS endpoint; failure to connect is a network issue, not a
code issue)
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import edge_tts

TICKS = 10_000_000  # 100-ns ticks per second (edge_tts.constants.TICKS_PER_SECOND)


async def main() -> None:
    out = Path(__file__).parent / "_out" / "pilot_edge_tts.mp3"
    out.parent.mkdir(exist_ok=True)
    comm = edge_tts.Communicate(
        "Edge TTS pilot one two three.", voice="en-US-AriaNeural",
        rate="+0%", pitch="+0Hz", boundary="WordBoundary")
    size, words = 0, []
    buf = bytearray()
    async for msg in comm.stream():
        if msg["type"] == "audio":
            buf.extend(msg["data"])
            size = len(buf)
        elif msg["type"] == "WordBoundary":
            words.append((round(msg["offset"] / TICKS, 2),
                          round((msg["offset"] + msg["duration"]) / TICKS, 2),
                          msg["text"]))
    out.write_bytes(bytes(buf))  # disk I/O after the async loop
    print(f"mp3 bytes={size} word_events={len(words)}")
    print("first words:", words[:4])
    assert size > 1000 and len(words) >= 5, "unexpected edge-tts output"
    print("PASS")


if __name__ == "__main__":
    asyncio.run(main())
