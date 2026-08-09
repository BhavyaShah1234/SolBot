"""Configuration loading.

The entire system is driven by ``config.yaml``. This module is the only place
that reads it, so nothing else in the codebase needs to know where the file
lives or how ``${ENV_VAR}`` placeholders are resolved.

Public surface
--------------
Config          Thin wrapper over the parsed dictionary with dotted lookup.
load_config()   Read, expand and validate the YAML file. Cached per path.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any) -> Any:
    """Recursively replace ``${VAR}`` with ``os.environ['VAR']``.

    Missing variables collapse to an empty string rather than raising, so a
    developer can run the read-only parts of the system (retrieval, planning)
    without Confluence credentials configured.
    """
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


class Config:
    """Read-only view over the configuration tree.

    ``cfg.get("retrieval.final_k")`` is preferred over ``cfg.data["retrieval"]
    ["final_k"]`` because it survives a missing intermediate key and lets every
    call site declare its own default.
    """

    def __init__(self, data: dict) -> None:
        self.data = data

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Look up a value by dot-separated path, e.g. ``"llm.chat_model"``."""
        node: Any = self.data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict:
        """Return one top-level block as a plain dictionary."""
        return dict(self.get(name, {}) or {})

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"Config(sections={sorted(self.data)})"


_REQUIRED_SECTIONS = (
    "llm",
    "vector_store",
    "retrieval",
    "planner",
    "worker",
)


@lru_cache(maxsize=8)
def load_config(path: str = "config.yaml") -> Config:
    """Parse ``path`` into a :class:`Config`.

    Results are cached by path so that importing modules can call this freely
    without re-reading the file on every request.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    raw = _expand(raw)

    missing = [name for name in _REQUIRED_SECTIONS if name not in raw]
    if missing:
        raise ValueError(f"config.yaml is missing required sections: {missing}")

    return Config(raw)
