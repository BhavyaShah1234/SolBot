"""Minimal, standalone config loader for the web_search package.

Deliberately does not import from ``db_engine`` — this package has no
dependency on it or on any orchestrator. A few lines of YAML-loading are
duplicated from ``db_engine.config.read_config`` rather than sharing it, so
``web_search`` stays independently usable, testable, and committable.
"""

import yaml

_DEFAULTS = {
    "enabled": True,
    "search_provider": "ddgs",
    "extraction_backend": "trafilatura",
    "max_results": 5,
    "search_timeout_seconds": 10,
    "fetch_timeout_seconds": 15,
    "max_fetch_chars": 12000,
    "min_extracted_chars": 200,
    "blocked_domains": [],
}


def read_config(path: str = "config.yaml") -> dict:
    """Loads the YAML configuration file and returns its ``web`` section.

    Args:
        path: Filesystem path to the YAML config file, relative to the
            current working directory unless absolute.

    Returns:
        The ``web`` section as a dict, merged over :data:`_DEFAULTS` so
        every key this package reads is always present even if
        ``config.yaml`` only overrides a few of them. Returns the defaults
        unchanged if the file has no top-level ``web`` key.

    Raises:
        FileNotFoundError: If no file exists at ``path``.
        yaml.YAMLError: If the file exists but is not valid YAML.
    """
    with open(path, "r") as f:
        full_cfg = yaml.safe_load(f) or {}
    return {**_DEFAULTS, **full_cfg.get("web", {})}
