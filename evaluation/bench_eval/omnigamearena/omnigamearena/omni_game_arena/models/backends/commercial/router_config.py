"""Read commercial backend endpoint settings from configs/router.yaml."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _router_path() -> Path:
    override = os.getenv("OMNI_ARENA_ROUTER_CONFIG")
    if override:
        path = Path(override).expanduser()
        return path if path.is_absolute() else (_repo_root() / path)
    return _repo_root() / "configs" / "router.yaml"


@lru_cache(maxsize=4)
def _load_router(path: str) -> dict[str, Any]:
    router_path = Path(path)
    if not router_path.exists():
        raise RuntimeError(f"Router config not found: {router_path}")
    return yaml.safe_load(router_path.read_text(encoding="utf-8")) or {}


_ENV_FIELDS = {
    ("openai", "base_url"): (
        "OPENAI_BASE_URL",
    ),
    ("openai", "api_key"): (
        "OPENAI_API_KEY",
    ),
    ("anthropic", "base_url"): (
        "ANTHROPIC_BASE_URL",
    ),
    ("anthropic", "api_key"): (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
    ),
}


def _looks_placeholder(value: str | None) -> bool:
    if value in (None, ""):
        return True
    text = str(value).strip()
    return text.startswith("<") and text.endswith(">")


def commercial_value(route_name: str, field: str) -> str:
    """Return a commercial route field from env, then configs/router.yaml.

    Job env (OPENAI_API_KEY / ANTHROPIC_BASE_URL / ...) wins so a
    shared router.yaml can keep placeholders without baking keys into git.
    """
    for env_name in _ENV_FIELDS.get((route_name, field), ()):
        env_val = os.getenv(env_name)
        if env_val and not _looks_placeholder(env_val):
            return env_val.strip()

    router_path = _router_path().resolve()
    router = _load_router(str(router_path))
    route = ((router.get("commercial") or {}).get(route_name) or {})
    if not isinstance(route, dict):
        route = {}

    value = route.get(field)
    if _looks_placeholder(value if isinstance(value, str) else None):
        raise RuntimeError(
            f"Missing commercial.{route_name}.{field} in {router_path} "
            f"and no env fallback ({', '.join(_ENV_FIELDS.get((route_name, field), ()))})"
        )
    return str(value)
