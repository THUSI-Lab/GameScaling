"""Agent factory and lazy export map for the agents package."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

from .mm_agents.base.base_client import BaseClient

_MM_AGENTS_DIR = Path(__file__).resolve().parent / "mm_agents"


_MODEL_CATALOG = {
    f"qwen3-vl-8b-sft-g{i}": {
        "module": "qwen_3_vl_sft",
        "client": "Qwen3VLSFTAgent",
        "config": "Qwen3VLSFTConfig",
    }
    for i in range(1, 9)
}


def get_config_for_model(model_id: str):
    catalog_entry = _MODEL_CATALOG.get(_require_model_id(model_id))
    if catalog_entry:
        config_cls = _load_symbol(catalog_entry["module"], catalog_entry["config"])
        return config_cls()
    raise ValueError(f"Unknown model id: {model_id}.")


def create_client(
    model_id: str,
    config,
    **kwargs,
) -> BaseClient:
    """Create a client instance for a registered model id."""
    catalog_entry = _MODEL_CATALOG.get(_require_model_id(model_id))
    client_cls = (
        _load_symbol(catalog_entry["module"], catalog_entry["client"])
        if catalog_entry
        else None
    )
    if not client_cls:
        raise ValueError(
            f"Unknown model id: {model_id}. Supported model ids: {', '.join(_MODEL_CATALOG.keys())}"
        )
    return client_cls(config, **kwargs)


def _require_model_id(model_id: str) -> str:
    text = str(model_id).strip()
    if not text:
        raise ValueError("Model id cannot be empty.")
    return text


_SYMBOL_EXPORTS: dict[str, tuple[str, str]] = {}
for _entry in _MODEL_CATALOG.values():
    for _kind in ("client", "config"):
        _SYMBOL_EXPORTS[_entry[_kind]] = (_entry["module"], _entry[_kind])


def _load_model_module(module_stem: str):
    module_name = f"{__package__}.mm_agents._model_file_{module_stem}"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    module_path = _MM_AGENTS_DIR / f"{module_stem}.py"
    spec = spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load agent module from {module_path}")

    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_symbol(module_stem: str, attr_name: str) -> Any:
    module = _load_model_module(module_stem)
    return getattr(module, attr_name)


def load_registered_symbol(name: str) -> Any:
    target = _SYMBOL_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_stem, attr_name = target
    value = _load_symbol(module_stem, attr_name)
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    return load_registered_symbol(name)


__all__ = sorted({"create_client", "get_config_for_model", *list(_SYMBOL_EXPORTS)})
