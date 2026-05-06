# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Seedance Video Generation Service - Volcengine Ark API client

Calls ByteDance's Seedance video generation models via Volcengine Ark
(https://ark.cn-beijing.volces.com/api/v3). Uses async polling pattern:
POST create task -> GET poll status -> return remote video URL.

The returned MediaResult.url points to a remote URL that expires in 24h;
downstream frame_processor._download_media will fetch it immediately.
"""

import asyncio
import math
import time
from math import gcd
from typing import Any, Dict, Optional, Tuple

import httpx
from loguru import logger

from pixelle_video.models.media import MediaResult


class SeedanceGenerationError(Exception):
    """Raised when Seedance video generation fails."""


_RATIO_MAP = {
    (9, 16): "9:16",
    (16, 9): "16:9",
    (1, 1): "1:1",
    (4, 3): "4:3",
    (3, 4): "3:4",
    (21, 9): "21:9",
}


class SeedanceVideoClient:
    """
    Volcengine Ark Seedance video generation client.

    Endpoints:
        POST {base_url}/contents/generations/tasks         -> create job
        GET  {base_url}/contents/generations/tasks/{id}    -> poll status

    Volcengine convention: resolution / duration / ratio / watermark / seed
    are passed as inline `--key value` parameters appended to the prompt text.
    """

    DEFAULT_MODEL = "doubao-seedance-2-0-260128"
    DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

    def __init__(self, config: Dict[str, Any]):
        self.api_key = (config.get("api_key") or "").strip()
        if not self.api_key:
            raise ValueError(
                "Volcengine ARK API key not configured. "
                "Set comfyui.seedance.api_key in config.yaml"
            )
        self.base_url = (config.get("base_url") or self.DEFAULT_BASE_URL).rstrip("/")
        self.poll_interval = float(config.get("poll_interval", 5.0))
        self.total_timeout = float(config.get("timeout", 600.0))
        self.request_timeout = float(config.get("request_timeout", 30.0))

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        prompt: str,
        provider_config: Optional[Dict[str, Any]] = None,
        duration: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> MediaResult:
        """
        Generate a video end-to-end (create job -> poll -> return remote URL).

        Args:
            prompt: Text prompt for video generation.
            provider_config: Per-workflow defaults from the JSON metadata file
                (model, resolution, duration, ratio, watermark, seed).
            duration: Optional runtime override (seconds, may be float). Will be
                ceiled and clamped to [4, 15] to fit Seedance 2.0 constraints.
            width / height: Optional runtime size override; used to derive ratio.

        Returns:
            MediaResult(media_type="video", url=<remote URL>, duration=<int seconds>).
        """
        cfg = provider_config or {}
        model = cfg.get("model", self.DEFAULT_MODEL)
        seconds = self._resolve_duration(cfg, duration)
        ratio = self._resolve_ratio(cfg, width, height)
        resolution = cfg.get("resolution", "720p")
        watermark = "true" if cfg.get("watermark", False) else "false"

        # Volcengine convention: parameters appended as `--key value` to the prompt
        full_prompt = (
            f"{prompt} --resolution {resolution} --duration {seconds} "
            f"--ratio {ratio} --watermark {watermark}"
        )
        if "seed" in cfg:
            full_prompt += f" --seed {cfg['seed']}"

        payload = {
            "model": model,
            "content": [{"type": "text", "text": full_prompt}],
        }

        logger.warning(
            "[Seedance] Video generation typically takes 30-120s; "
            "use /api/video/generate/async for long-running calls."
        )
        logger.info(
            f"[Seedance] Submitting task: model={model} duration={seconds}s "
            f"resolution={resolution} ratio={ratio}"
        )

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.request_timeout),
            base_url=self.base_url,
            headers=self._headers,
        ) as client:
            task_id = await self._create_task(client, payload)
            video_url = await self._poll_until_done(client, task_id)

        logger.info(f"[Seedance] Generated video: {video_url}")
        return MediaResult(media_type="video", url=video_url, duration=float(seconds))

    async def _create_task(self, client: httpx.AsyncClient, payload: Dict[str, Any]) -> str:
        try:
            resp = await client.post("/contents/generations/tasks", json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise SeedanceGenerationError(
                f"Seedance create failed [{e.response.status_code}]: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise SeedanceGenerationError(f"Seedance create network error: {e}") from e

        task = resp.json()
        task_id = task.get("id")
        if not task_id:
            raise SeedanceGenerationError(f"Seedance response missing task id: {task}")
        logger.debug(f"[Seedance] Created task: {task_id}")
        return task_id

    async def _poll_until_done(self, client: httpx.AsyncClient, task_id: str) -> str:
        deadline = time.monotonic() + self.total_timeout
        attempt = 0
        while True:
            if time.monotonic() > deadline:
                raise SeedanceGenerationError(
                    f"Seedance task {task_id} timed out after {self.total_timeout}s"
                )
            attempt += 1
            try:
                resp = await client.get(f"/contents/generations/tasks/{task_id}")
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                # 5xx: retry with bounded backoff; 4xx: terminal
                if 500 <= e.response.status_code < 600 and attempt < 5:
                    await asyncio.sleep(min(self.poll_interval * attempt, 30))
                    continue
                raise SeedanceGenerationError(
                    f"Seedance poll failed [{e.response.status_code}]: {e.response.text}"
                ) from e
            except httpx.RequestError as e:
                logger.warning(f"[Seedance] poll network glitch (attempt={attempt}): {e}")
                await asyncio.sleep(self.poll_interval)
                continue

            data = resp.json()
            status = data.get("status")
            logger.debug(f"[Seedance] {task_id} status={status} (attempt={attempt})")

            if status == "succeeded":
                content = data.get("content") or {}
                url = content.get("video_url") or data.get("video_url")
                if not url:
                    raise SeedanceGenerationError(
                        f"Seedance task succeeded but no video_url in response: {data}"
                    )
                return url
            if status in ("failed", "cancelled", "error"):
                err = data.get("error") or data.get("failure_reason") or "unknown"
                raise SeedanceGenerationError(
                    f"Seedance task {task_id} {status}: {err}"
                )
            # status in {queued, running, ...} -> keep waiting
            await asyncio.sleep(self.poll_interval)

    @staticmethod
    def _resolve_duration(cfg: Dict[str, Any], duration: Optional[float]) -> int:
        """Ceil + clamp to Seedance 2.0 valid range [4, 15] integer seconds."""
        target = duration if duration else cfg.get("duration", 5)
        return max(4, min(15, math.ceil(float(target))))

    @staticmethod
    def _resolve_ratio(
        cfg: Dict[str, Any],
        width: Optional[int],
        height: Optional[int],
    ) -> str:
        """Derive Seedance ratio token from width/height; fall back to cfg or 16:9."""
        if width and height:
            g = gcd(int(width), int(height))
            key: Tuple[int, int] = (int(width) // g, int(height) // g)
            mapped = _RATIO_MAP.get(key)
            if mapped:
                return mapped
        return cfg.get("ratio", "16:9")
