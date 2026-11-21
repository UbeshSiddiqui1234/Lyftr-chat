"""Application settings, read entirely from environment variables."""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

DEFAULT_MODEL = "gemini/gemini-3.5-flash-lite"
DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise assistant."

_KEY_VAR = re.compile(r"LLM_KEY_([1-9]\d*)")
_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ProviderKey:
    id: str
    model: str
    api_key: str = field(repr=False)
    api_base: str | None = None


@dataclass(frozen=True)
class Settings:
    database_path: str
    keys: tuple[ProviderKey, ...]
    cors_origins: tuple[str, ...]
    enable_failure_simulation: bool = False
    cache_enabled: bool = True
    cache_type: str = "disk"
    cache_dir: str = "/data/llm-cache"
    cache_ttl_seconds: int = 86400
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT
    llm_timeout_seconds: float = 25.0
    llm_max_output_tokens: int = 2048
    max_history_messages: int = 100


def sqlite_path_from_url(url: str) -> str:
    """Convert sqlite:////data/app.db into a filesystem path."""
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        raise ValueError("DATABASE_URL must start with sqlite:///")
    path = url[len(prefix) :]
    if not path or path == ":memory:":
        raise ValueError("DATABASE_URL must point to a database file")
    return path


def load_key_pool(env: Mapping[str, str]) -> tuple[ProviderKey, ...]:
    """Read LLM_KEY_<n>, with optional LLM_MODEL_<n> and LLM_API_BASE_<n>."""
    default_model = env.get("LLM_MODEL", "").strip() or DEFAULT_MODEL
    numbers = sorted(
        int(match.group(1))
        for name, value in env.items()
        if (match := _KEY_VAR.fullmatch(name)) and value.strip()
    )
    return tuple(
        ProviderKey(
            id=f"key_{n}",
            api_key=env[f"LLM_KEY_{n}"].strip(),
            model=env.get(f"LLM_MODEL_{n}", "").strip() or default_model,
            api_base=env.get(f"LLM_API_BASE_{n}", "").strip() or None,
        )
        for n in numbers
    )


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name, "").strip().lower()
    return value in _TRUE_VALUES if value else default


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    origins = env.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    cache_type = env.get("CACHE_TYPE", "disk").strip().lower()
    if cache_type not in {"disk", "local"}:
        raise ValueError("CACHE_TYPE must be disk or local")
    return Settings(
        database_path=sqlite_path_from_url(env.get("DATABASE_URL", "sqlite:////data/app.db")),
        keys=load_key_pool(env),
        cors_origins=tuple(o.strip().rstrip("/") for o in origins.split(",") if o.strip()),
        enable_failure_simulation=_flag(env, "ENABLE_FAILURE_SIMULATION", False),
        cache_enabled=_flag(env, "CACHE_ENABLED", True),
        cache_type=cache_type,
        cache_dir=env.get("CACHE_DIR", "/data/llm-cache"),
        cache_ttl_seconds=int(env.get("CACHE_TTL_SECONDS", "86400")),
        default_system_prompt=env.get("DEFAULT_SYSTEM_PROMPT", "").strip() or DEFAULT_SYSTEM_PROMPT,
        llm_timeout_seconds=float(env.get("LLM_TIMEOUT_SECONDS", "25")),
        llm_max_output_tokens=int(env.get("LLM_MAX_OUTPUT_TOKENS", "2048")),
        max_history_messages=int(env.get("MAX_HISTORY_MESSAGES", "100")),
    )
