import asyncio
from pathlib import Path

import edge_tts


async def synth_one(text: str, voice: str, out: Path,
                    timeout_s: float = 60) -> None:
    comm = edge_tts.Communicate(text, voice=voice,
                                boundary="WordBoundary")
    audio = bytearray()

    async def _pull():
        async for msg in comm.stream():
            if msg["type"] == "audio":
                audio.extend(msg["data"])

    await asyncio.wait_for(_pull(), timeout=timeout_s)
    if not audio:
        raise RuntimeError("tts returned no audio for this text")
    out.write_bytes(bytes(audio))
