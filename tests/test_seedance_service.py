# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Unit tests for SeedanceVideoClient.

Uses httpx.MockTransport to avoid real network calls. Covers:
- Successful create -> poll -> succeeded path with video_url extraction
- Polling iterates through queued -> running -> succeeded
- Failure status (failed/cancelled) raises SeedanceGenerationError
- 5xx server error retries up to 5 times then raises
- Total timeout enforcement
- Duration round-up + clamp to [4, 15]
- Ratio derivation from width/height
- Missing API key raises ValueError
"""

from unittest.mock import patch

import httpx
import pytest

from pixelle_video.services.seedance_service import (
    SeedanceGenerationError,
    SeedanceVideoClient,
)


def _client_with_transport(transport: httpx.MockTransport, **overrides) -> SeedanceVideoClient:
    """Build a client and patch httpx.AsyncClient to use a MockTransport."""
    cfg = {
        "api_key": "test-key",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "poll_interval": 0.0,
        "timeout": 5.0,
        "request_timeout": 5.0,
    }
    cfg.update(overrides)
    return SeedanceVideoClient(cfg), transport


class _PatchAsyncClient:
    """Context manager that swaps httpx.AsyncClient to use a fixed transport."""

    def __init__(self, transport: httpx.MockTransport):
        self.transport = transport
        self._real = httpx.AsyncClient

    def __enter__(self):
        transport = self.transport

        class _Patched(self._real):
            def __init__(self, *a, **kw):
                kw["transport"] = transport
                super().__init__(*a, **kw)

        self._patcher = patch("pixelle_video.services.seedance_service.httpx.AsyncClient", _Patched)
        self._patcher.start()
        return self

    def __exit__(self, *exc):
        self._patcher.stop()


# ==================== Tests: configuration ====================

def test_missing_api_key_raises():
    with pytest.raises(ValueError, match="API key not configured"):
        SeedanceVideoClient({"api_key": ""})


def test_default_base_url():
    c = SeedanceVideoClient({"api_key": "k"})
    assert c.base_url == "https://ark.cn-beijing.volces.com/api/v3"


# ==================== Tests: duration / ratio resolvers ====================

@pytest.mark.parametrize(
    "duration,expected",
    [
        (None, 5),    # falls back to cfg default 5
        (3.0, 4),     # below floor
        (4.0, 4),     # exact floor
        (4.5, 5),     # ceil
        (8.0, 8),
        (9.5, 10),    # ceil
        (15.0, 15),   # exact ceiling
        (20.0, 15),   # clamp
    ],
)
def test_resolve_duration(duration, expected):
    cfg = {"duration": 5}
    assert SeedanceVideoClient._resolve_duration(cfg, duration) == expected


@pytest.mark.parametrize(
    "w,h,expected",
    [
        (1080, 1920, "9:16"),
        (1920, 1080, "16:9"),
        (1080, 1080, "1:1"),
        (1024, 768, "4:3"),
        (768, 1024, "3:4"),
        (None, None, "16:9"),  # fallback to cfg default
        (1080, 1234, "16:9"),  # non-standard -> fallback
    ],
)
def test_resolve_ratio(w, h, expected):
    cfg = {"ratio": "16:9"}
    assert SeedanceVideoClient._resolve_ratio(cfg, w, h) == expected


# ==================== Tests: end-to-end flow with mocked transport ====================

@pytest.mark.asyncio
async def test_generate_success_polls_until_succeeded():
    poll_responses = iter(
        [
            httpx.Response(200, json={"id": "cgt-1", "status": "queued"}),
            httpx.Response(200, json={"id": "cgt-1", "status": "running"}),
            httpx.Response(
                200,
                json={
                    "id": "cgt-1",
                    "status": "succeeded",
                    "content": {"video_url": "https://cdn.example.com/v.mp4"},
                },
            ),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/contents/generations/tasks"):
            return httpx.Response(200, json={"id": "cgt-1"})
        if request.method == "GET" and "/contents/generations/tasks/cgt-1" in request.url.path:
            return next(poll_responses)
        return httpx.Response(404, json={"error": "unexpected"})

    transport = httpx.MockTransport(handler)
    client, _ = _client_with_transport(transport)
    with _PatchAsyncClient(transport):
        result = await client.generate(
            prompt="a dog runs through a meadow",
            provider_config={"model": "doubao-seedance-2-0-260128", "duration": 5},
            duration=4.5,
            width=1080,
            height=1920,
        )
    assert result.media_type == "video"
    assert result.url == "https://cdn.example.com/v.mp4"
    assert result.duration == 5.0  # ceil(4.5) = 5


@pytest.mark.asyncio
async def test_generate_failed_status_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"id": "cgt-2"})
        return httpx.Response(
            200, json={"id": "cgt-2", "status": "failed", "error": "model rejected prompt"}
        )

    transport = httpx.MockTransport(handler)
    client, _ = _client_with_transport(transport)
    with _PatchAsyncClient(transport):
        with pytest.raises(SeedanceGenerationError, match="failed"):
            await client.generate(prompt="x", provider_config={})


@pytest.mark.asyncio
async def test_create_4xx_raises_immediately():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad request"})

    transport = httpx.MockTransport(handler)
    client, _ = _client_with_transport(transport)
    with _PatchAsyncClient(transport):
        with pytest.raises(SeedanceGenerationError, match="create failed"):
            await client.generate(prompt="x", provider_config={})


@pytest.mark.asyncio
async def test_poll_5xx_retries_then_raises():
    state = {"posts": 0, "gets": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            state["posts"] += 1
            return httpx.Response(200, json={"id": "cgt-3"})
        state["gets"] += 1
        return httpx.Response(503, json={"error": "service unavailable"})

    transport = httpx.MockTransport(handler)
    client, _ = _client_with_transport(transport)
    with _PatchAsyncClient(transport):
        with pytest.raises(SeedanceGenerationError, match="poll failed"):
            await client.generate(prompt="x", provider_config={})

    # The poll loop retries on 5xx up to 5 attempts, then raises on the 6th
    assert state["gets"] == 5


@pytest.mark.asyncio
async def test_total_timeout_enforced():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"id": "cgt-4"})
        # always still in progress
        return httpx.Response(200, json={"id": "cgt-4", "status": "running"})

    transport = httpx.MockTransport(handler)
    client, _ = _client_with_transport(transport, timeout=0.05, poll_interval=0.01)
    with _PatchAsyncClient(transport):
        with pytest.raises(SeedanceGenerationError, match="timed out"):
            await client.generate(prompt="x", provider_config={})


@pytest.mark.asyncio
async def test_succeeded_without_url_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"id": "cgt-5"})
        return httpx.Response(200, json={"id": "cgt-5", "status": "succeeded", "content": {}})

    transport = httpx.MockTransport(handler)
    client, _ = _client_with_transport(transport)
    with _PatchAsyncClient(transport):
        with pytest.raises(SeedanceGenerationError, match="no video_url"):
            await client.generate(prompt="x", provider_config={})


@pytest.mark.asyncio
async def test_payload_uses_top_level_body_fields():
    """Volcengine Ark expects resolution/ratio/duration/watermark/seed as top-level
    JSON body fields, not inline `--key value` prompt suffixes (those are silently
    ignored by the official endpoint, which is what was causing watermarks to leak
    onto generated videos)."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            captured["body"] = request.read()
            return httpx.Response(200, json={"id": "cgt-6"})
        return httpx.Response(
            200,
            json={
                "id": "cgt-6",
                "status": "succeeded",
                "content": {"video_url": "https://cdn/v.mp4"},
            },
        )

    transport = httpx.MockTransport(handler)
    client, _ = _client_with_transport(transport)
    with _PatchAsyncClient(transport):
        await client.generate(
            prompt="cat playing piano",
            provider_config={
                "model": "doubao-seedance-2-0-fast-260128",
                "resolution": "480p",
                "duration": 4,
                "ratio": "1:1",
                "watermark": False,
                "seed": 42,
            },
        )

    import json as _json
    body = _json.loads(captured["body"].decode())
    assert body["model"] == "doubao-seedance-2-0-fast-260128"
    assert body["resolution"] == "480p"
    assert body["duration"] == 4
    assert body["ratio"] == "1:1"
    assert body["watermark"] is False
    assert body["seed"] == 42
    # Prompt should remain clean — no inline --key value suffixes
    assert body["content"][0]["text"] == "cat playing piano"
    assert "--watermark" not in body["content"][0]["text"]


@pytest.mark.asyncio
async def test_watermark_defaults_to_false_when_unset():
    """Default workflow JSON sets watermark=false; verify it propagates."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            captured["body"] = request.read()
            return httpx.Response(200, json={"id": "cgt-7"})
        return httpx.Response(
            200,
            json={
                "id": "cgt-7",
                "status": "succeeded",
                "content": {"video_url": "https://cdn/v.mp4"},
            },
        )

    transport = httpx.MockTransport(handler)
    client, _ = _client_with_transport(transport)
    with _PatchAsyncClient(transport):
        await client.generate(prompt="test", provider_config={})  # no watermark key

    import json as _json
    body = _json.loads(captured["body"].decode())
    assert body["watermark"] is False
