"""Configuration loading for ShardCoder.

Configuration is layered:

  defaults  <  shardcoder.toml  <  environment variables  <  CLI overrides

The result is a single :class:`ShardCoderConfig` Pydantic model that the
rest of the agent can rely on.

Environment variables use the prefix ``SHARDCODER_`` and a double-underscore
separator for nested keys, e.g. ``SHARDCODER_LLM__MODEL=gemma-4`` overrides
``[llm].model``.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    base_url: str = "http://localhost:1234/v1"
    api_key: str | None = None
    model: str = "auto"
    max_context_tokens: int = 8192
    max_output_tokens: int = 2048
    temperature: float = 0.1
    timeout_seconds: int = 300
    timeout_retries: int = Field(default=1, ge=0)
    timeout_retry_backoff_seconds: float = Field(default=1.0, ge=0.0)


class AgentConfig(BaseModel):
    max_iterations: int = 4
    dry_run: bool = False
    auto_apply: bool = True
    auto_run_tests: bool = True


class RepositoryConfig(BaseModel):
    ignore_dirs: list[str] = Field(
        default_factory=lambda: [
            ".git",
            "node_modules",
            "dist",
            "build",
            "target",
            ".venv",
            "__pycache__",
            ".next",
            "vendor",
            "coverage",
            ".mypy_cache",
            ".pytest_cache",
        ]
    )
    max_file_bytes: int = 200_000


class ValidationConfig(BaseModel):
    test_command: str = ""
    lint_command: str = ""
    format_command: str = ""


class ContextConfig(BaseModel):
    max_memory_tokens: int = 800
    max_snippet_tokens: int = 3500
    max_summary_tokens: int = 2000
    max_validation_tokens: int = 1000
    max_web_context_tokens: int = 800


class WebConfig(BaseModel):
    enabled: bool = False
    max_queries: int = 3
    max_results_per_query: int = 5
    max_fetched_pages: int = 2
    require_user_approval: bool = True
    prefer_official_docs: bool = True
    backend: str = "disabled"


class ShardCoderConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    repository: RepositoryConfig = Field(default_factory=RepositoryConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    web: WebConfig = Field(default_factory=WebConfig)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


CONFIG_FILENAMES = ("shardcoder.toml", ".shardcoder.toml")
ENV_PREFIX = "SHARDCODER_"


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or validated."""


def find_config_file(start: Path | str | None = None) -> Path | None:
    """Walk upward from *start* looking for a shardcoder config file."""
    here = Path(start or Path.cwd()).resolve()
    for candidate_dir in (here, *here.parents):
        for name in CONFIG_FILENAMES:
            candidate = candidate_dir / name
            if candidate.is_file():
                return candidate
    return None


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Failed to read config file {path}: {exc}") from exc


def _coerce_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", ""}:
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _env_overrides() -> dict[str, Any]:
    """Translate ``SHARDCODER_<SECTION>__<KEY>`` env vars into nested dict."""
    out: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = key[len(ENV_PREFIX) :].lower().split("__")
        if not path:
            continue
        cursor = out
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                # conflicting env vars; ignore
                cursor = {}
        cursor[path[-1]] = _coerce_scalar(value)
    return out


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    config_path: Path | str | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> ShardCoderConfig:
    """Load and validate config from disk, env vars, and CLI overrides."""
    data: dict[str, Any] = {}

    path = Path(config_path) if config_path else find_config_file()
    if path is not None and path.is_file():
        data = _deep_merge(data, _load_toml(path))

    data = _deep_merge(data, _env_overrides())
    if cli_overrides:
        data = _deep_merge(data, cli_overrides)

    try:
        return ShardCoderConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration: {exc}") from exc
