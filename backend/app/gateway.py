"""LiteLLM gateway: response cache, key pool and automatic failover."""

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

# Use the bundled model price map instead of fetching it at import time.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm  # noqa: E402
from litellm.caching.caching import Cache  # noqa: E402

from .config import ProviderKey, Settings  # noqa: E402

logger = logging.getLogger("gateway")

litellm.suppress_debug_info = True
litellm.telemetry = False
litellm.drop_params = True

LIVE_INFERENCE = "LIVE_INFERENCE"
LLM_CACHE = "LLM_CACHE"
SIMULATED_STATUSES = {429, 500, 502, 503}


class AllProvidersFailed(Exception):
    """Every key in the pool failed for this request."""


class EmptyReply(Exception):
    """The provider answered successfully but with no text."""


@dataclass(frozen=True)
class Completion:
    reply: str
    resolution_source: str
    model: str
    input_tokens: int
    output_tokens: int
    key_id: str | None
    fallback_attempts: int


def parse_simulation_header(header: str | None) -> dict[str, int]:
    """Parse 'key_1=429,key_2=500' or '*=429' into {key_id: status}."""
    if header is None or not header.strip():
        return {}
    failures: dict[str, int] = {}
    for pair in header.split(","):
        key_id, separator, status = (part.strip() for part in pair.partition("="))
        if not separator or not key_id or not status.isdecimal():
            raise ValueError("expected comma-separated <key_id>=<status> pairs")
        if int(status) not in SIMULATED_STATUSES:
            raise ValueError("status must be one of 429, 500, 502, 503")
        failures[key_id] = int(status)
    return failures


def _build_cache(settings: Settings) -> Cache | None:
    if not settings.cache_enabled:
        return None
    # max_messages=None: LiteLLM otherwise skips caching beyond four messages.
    options = {"ttl": settings.cache_ttl_seconds, "max_messages": None}
    if settings.cache_type == "disk":
        Path(settings.cache_dir).mkdir(parents=True, exist_ok=True)
        cache = Cache(type="disk", disk_cache_dir=settings.cache_dir, **options)
    else:
        cache = Cache(type="local", **options)
    # Reads and writes go through this object so a write finishes before the
    # response returns. Remove the background callbacks Cache() registers.
    litellm.disable_cache()
    return cache


class Gateway:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.keys = settings.keys
        self.cache = _build_cache(settings)
        self.models = list(dict.fromkeys(key.model for key in self.keys))

    @property
    def cache_enabled(self) -> bool:
        return self.cache is not None

    async def complete(
        self, messages: list[dict[str, str]], simulated: dict[str, int] | None = None
    ) -> Completion:
        simulated = simulated or {}
        # A simulated failure must reach the key pool, so it skips the cache read.
        if not simulated:
            cached = await self._read_cache(messages)
            if cached is not None:
                return cached

        failed = 0
        for key in self.keys:
            status = simulated.get(key.id, simulated.get("*"))
            if status is not None:
                logger.warning("%s simulated failure status=%s", key.id, status)
                failed += 1
                continue
            try:
                completion = await self._call(key, messages, failed)
            except Exception as exc:  # Any provider failure moves to the next key.
                # Provider messages can embed the API key, so log only safe fields.
                logger.warning(
                    "%s failed error=%s status=%s",
                    key.id,
                    type(exc).__name__,
                    getattr(exc, "status_code", None),
                )
                failed += 1
                continue
            await self._write_cache(messages, completion)
            return completion
        raise AllProvidersFailed

    async def _call(
        self, key: ProviderKey, messages: list[dict[str, str]], failed: int
    ) -> Completion:
        response = await litellm.acompletion(
            model=key.model,
            messages=[dict(message) for message in messages],
            api_key=key.api_key,
            api_base=key.api_base,
            timeout=self.settings.llm_timeout_seconds,
            max_tokens=self.settings.llm_max_output_tokens,
            num_retries=0,
            max_retries=0,
            caching=False,
        )
        reply = (response.choices[0].message.content or "").strip()
        if not reply:
            raise EmptyReply
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        if input_tokens is None:
            input_tokens = litellm.token_counter(model=key.model, messages=messages)
        if output_tokens is None:
            output_tokens = litellm.token_counter(model=key.model, text=reply)
        return Completion(
            reply=reply,
            resolution_source=LIVE_INFERENCE,
            model=key.model,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            key_id=key.id,
            fallback_attempts=failed,
        )

    async def _read_cache(self, messages: list[dict[str, str]]) -> Completion | None:
        if self.cache is None:
            return None
        for model in self.models:
            entry = await asyncio.to_thread(self.cache.get_cache, model=model, messages=messages)
            if isinstance(entry, dict) and entry.get("reply"):
                return Completion(
                    reply=entry["reply"],
                    resolution_source=LLM_CACHE,
                    model=model,
                    input_tokens=int(entry["input_tokens"]),
                    output_tokens=int(entry["output_tokens"]),
                    key_id=None,
                    fallback_attempts=0,
                )
        return None

    async def _write_cache(self, messages: list[dict[str, str]], completion: Completion) -> None:
        if self.cache is None:
            return
        entry = {
            "reply": completion.reply,
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
        }
        await asyncio.to_thread(
            self.cache.add_cache, entry, model=completion.model, messages=messages
        )
