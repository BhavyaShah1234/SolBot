"""Configuration loading for the Memory Engine.

Reads tunable values from ``config.yaml`` (SQLite path, session/user
defaults, extraction and recall thresholds, etc.) under its own top-level
``memory`` key. This package has no secrets of its own -- Confluence
credentials belong to ``db_engine``, and any future LLM-provider
credentials belong to the ``agents`` package that will eventually call
this package's primitives.

Deliberately does not import from ``db_engine`` -- like ``web_search``,
this package duplicates a few lines of YAML-loading rather than sharing
``db_engine.config.read_config``, so it stays independently usable,
testable, and committable.
"""

import yaml


def read_config(path: str = "config.yaml") -> dict:
    """Loads the YAML configuration file into a plain dict.

    Args:
        path: Path to the YAML config file, relative to the current working
            directory unless absolute.

    Returns:
        The parsed configuration as a plain dict (empty dict if the file is
        empty). Callers index into ``cfg["memory"][...]`` for this
        package's own settings, and ``cfg["embedding"]["model"]`` for the
        embedding model ``recall.py`` shares with ``db_engine``.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}
