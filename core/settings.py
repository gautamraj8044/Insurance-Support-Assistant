"""
core/settings.py - Centralised, typed application configuration.

All environment variables and tunable constants are read here ONCE.
Nothing else in the codebase should call os.getenv() directly - import
from this module instead. This is what makes the LLM model, DB paths,
and limits configurable without touching code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class GroqSettings:
    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    # Separate, smaller model for the lightweight intent classifier.
    # Defaults to the same model as `model` unless overridden.
    intent_model: str = field(
        default_factory=lambda: os.getenv("GROQ_INTENT_MODEL", "")
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("GROQ_TEMPERATURE", "0"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("GROQ_MAX_RETRIES", "2"))
    )
    request_timeout: int = field(
        default_factory=lambda: int(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))
    )

    def __post_init__(self):
        if not self.intent_model:
            object.__setattr__(self, "intent_model", self.model)


@dataclass(frozen=True)
class PathSettings:
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "data/insurance_support.db"))
    chroma_path: str = field(default_factory=lambda: os.getenv("CHROMA_PATH", "./chroma_db"))
    chroma_collection: str = "insurance_faq_collection"
    log_dir: str = field(default_factory=lambda: os.getenv("LOG_DIR", "logs"))


@dataclass(frozen=True)
class FAQSettings:
    dataset_name: str = field(
        default_factory=lambda: os.getenv("FAQ_DATASET", "deccan-ai/insuranceQA-v2")
    )
    sample_size: int = field(default_factory=lambda: int(os.getenv("FAQ_SAMPLE_SIZE", "500")))
    batch_size: int = field(default_factory=lambda: int(os.getenv("FAQ_BATCH_SIZE", "100")))
    top_k: int = field(default_factory=lambda: int(os.getenv("FAQ_TOP_K", "3")))


@dataclass(frozen=True)
class AppSettings:
    app_title: str = "Insurance Support Assistant"
    max_graph_iterations: int = field(
        default_factory=lambda: int(os.getenv("MAX_GRAPH_ITERATIONS", "4"))
    )
    debug_mode: bool = field(default_factory=lambda: _get_bool("DEBUG_MODE", False))
    enable_tracing: bool = field(default_factory=lambda: _get_bool("ENABLE_TRACING", False))
    phoenix_api_key: str = field(default_factory=lambda: os.getenv("PHOENIX_API_KEY", ""))
    phoenix_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
        )
    )
    phoenix_project: str = field(
        default_factory=lambda: os.getenv("PHOENIX_PROJECT_NAME", "insurance-support-assistant")
    )

    @property
    def phoenix_is_local(self) -> bool:
        """
        True if NOT pointed at Phoenix Cloud (i.e. self-hosted, no API key
        needed). Rather than trying to enumerate every possible local
        hostname (localhost, 127.0.0.1, a Docker Compose service name like
        "phoenix", a LAN IP, etc.), this checks for the one hostname that
        specifically IS Phoenix Cloud and treats everything else as local/
        self-hosted.
        """
        return "app.phoenix.arize.com" not in self.phoenix_endpoint.lower()


groq = GroqSettings()
paths = PathSettings()
faq = FAQSettings()
app = AppSettings()


def validate_settings() -> list[str]:
    """Return a list of human-readable problems with the current config."""
    problems: list[str] = []
    if not groq.api_key:
        problems.append(
            "GROQ_API_KEY is not set. Create a .env file with GROQ_API_KEY=<your key>."
        )
    if app.enable_tracing and not app.phoenix_api_key and not app.phoenix_is_local:
        problems.append(
            "ENABLE_TRACING is true but PHOENIX_API_KEY is not set. Get a Phoenix "
            "Cloud API key from https://app.phoenix.arize.com (Settings > API Keys) "
            "and set PHOENIX_API_KEY in your .env file, or point "
            "PHOENIX_COLLECTOR_ENDPOINT at a local Phoenix instance instead."
        )
    return problems
