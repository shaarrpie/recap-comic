import asyncio
from pathlib import Path

import edge_tts


async def synth_one(text: str, voice: str, out: Path,
                    rate: str = "", pitch: str = "",
                    timeout_s: float = 60) -> None:
    kwargs = {"boundary": "WordBoundary"}
    if rate:
        kwargs["rate"] = rate
    if pitch:
        kwargs["pitch"] = pitch
    # edge_tts.Communicate's signature mixes Literal/optional params, so a
    # conditional **kwargs is the only way to forward rate/pitch without
    # four near-duplicate call sites.
    comm = edge_tts.Communicate(text, voice=voice, **kwargs)  # type: ignore[arg-type]
    audio = bytearray()

    async def _pull():
        async for msg in comm.stream():
            if msg["type"] == "audio":
                audio.extend(msg["data"])

    await asyncio.wait_for(_pull(), timeout=timeout_s)
    if not audio:
        raise RuntimeError("tts returned no audio for this text")
    # Atomic write: an interrupted synthesis used to leave a partial mp3 on
    # disk that later callers accepted unconditionally (they only check
    # is_file()), serving corrupt audio forever after.
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(bytes(audio))
    tmp.replace(out)
