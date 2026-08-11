"""Configuration loading for the DB Engine.

Reads tunable values from ``config.yaml`` (chunk sizes, model names, vector
store paths, retrieval thresholds, etc.) and Confluence credentials from
environment variables loaded via ``.env``. Credentials never live in
``config.yaml`` — only in the environment.
"""

import os

import yaml
from dotenv import load_dotenv

load_dotenv()


def read_config(path: str = "config.yaml") -> dict:
    """Loads the YAML configuration file into a plain dict.

    Args:
        path: Filesystem path to the YAML config file, relative to the
            current working directory unless absolute.

    Returns:
        The parsed configuration as a nested dict. Returns an empty dict
        if the file exists but is empty.

    Raises:
        FileNotFoundError: If no file exists at ``path``.
        yaml.YAMLError: If the file exists but is not valid YAML.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def require_confluence_credentials() -> tuple[str, str]:
    """Reads and validates Confluence credentials from the environment.

    Returns:
        A ``(username, api_token)`` tuple.

    Raises:
        RuntimeError: If either ``CONFLUENCE_USERNAME`` or
            ``CONFLUENCE_API_TOKEN`` is unset in the environment.
    """
    username = os.environ.get("CONFLUENCE_USERNAME")
    token = os.environ.get("CONFLUENCE_API_TOKEN")
    if not username or not token:
        raise RuntimeError(
            "CONFLUENCE_USERNAME / CONFLUENCE_API_TOKEN not set — "
            "copy .env.example to .env and fill them in."
        )
    return username, token
