# tests/test_cloudflare_backend.py
"""Offline mock-based tests for the Cloudflare Workers AI backend.

Verifies the Llama 3.2 Vision request shape and the required license-
acceptance flow, without hitting the real API.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from test_guided import make_strip

import strip_analyzer as sa


def _make_backend():
    b = sa.CloudflareWorkersAIBackend.__new__(sa.CloudflareWorkersAIBackend)
    b.model = sa.CloudflareWorkersAIBackend.DEFAULT_MODEL
    b._api_key = "fake-token"
    b._account_id = "fake-account"
    b.usage_log = []
    b.last_usage = None
    b._agreed_to_license = False
    return b


def _post_side_effect(url: str, payload: dict, headers: dict) -> dict:
    if payload.get("prompt") == "agree":
        return {"success": True}
    if "messages" in payload and isinstance(payload["messages"], list):
        content = payload["messages"][0].get("content", [])
        if isinstance(content, list) and any(
                c.get("type") == "image_url" for c in content):
            return {
                "success": True,
                "result": {
                    "response": json.dumps({
                        "panels": [
                            {"panel_index": 1, "y_start": 100, "y_end": 500,
                             "narration": "ok", "dialogue": "", "panel_type": "single",
                             "confidence": 0.9, "bubble_boxes": []}
                        ]
                    })
                }
            }
    raise ValueError(f"unexpected payload shape: {payload}")


def test_cloudflare_default_model_is_llama() -> None:
    """The default model requires a license-agreement pre-request."""
    assert sa.CloudflareWorkersAIBackend.DEFAULT_MODEL == \
        "@cf/meta/llama-3.2-11b-vision-instruct"


def test_cloudflare_license_agreement_flow() -> None:
    """The backend sends a license-agreement payload before the real request."""
    backend = _make_backend()
    backend._post = MagicMock(side_effect=_post_side_effect)

    img = make_strip(600, panels=[(100, 500)], gutters=[])
    entries, chars = backend.analyze_chunk(img)

    assert backend._agreed_to_license is True
    assert len(entries) == 1
    assert entries[0].narration == "ok"
    assert backend._post.call_count == 2  # agree + real request


def test_cloudflare_llama_payload_shape() -> None:
    """Llama 3.2 Vision expects VLM-style content arrays with image_url parts."""
    backend = _make_backend()
    captured: list[dict] = []

    def fake_post(url, payload, headers):
        captured.append(payload)
        return _post_side_effect(url, payload, headers)

    backend._post = fake_post

    img = make_strip(600, panels=[(100, 500)], gutters=[])
    backend.analyze_chunk(img)

    real_payload = captured[1]  # second call is the real vision request
    assert "messages" in real_payload
    msgs = real_payload["messages"]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_cloudflare_skips_license_on_second_call() -> None:
    """Once the license is accepted, subsequent calls skip the agree step."""
    backend = _make_backend()
    backend._post = MagicMock(side_effect=_post_side_effect)

    img = make_strip(600, panels=[(100, 500)], gutters=[])
    backend.analyze_chunk(img)
    assert backend._post.call_count == 2

    backend.analyze_chunk(img)
    assert backend._post.call_count == 3  # agree skipped, only real request
