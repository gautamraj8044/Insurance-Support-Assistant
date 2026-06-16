"""
core/exceptions.py - Application-specific exception hierarchy.
"""


class InsuranceAgentError(Exception):
    """Base class for all application-specific errors."""


class LLMUnavailableError(InsuranceAgentError):
    """Raised when the LLM provider cannot be configured (e.g. missing API key)."""


class DatabaseError(InsuranceAgentError):
    """Raised when a database operation fails unexpectedly."""


class VectorStoreError(InsuranceAgentError):
    """Raised when the vector store (ChromaDB) is unavailable or misconfigured."""


class ConfigurationError(InsuranceAgentError):
    """Raised when required configuration is missing or invalid."""
