"""
graph/entity_extraction.py - Lightweight regex-based ID extraction.

Pulls policy numbers, customer IDs, and claim IDs out of free text so
the graph can carry them across turns without an extra LLM call.
Patterns match the synthetic data generator's ID formats
(see data/generator.py): POL000001, CUST00001, CLM000001.
"""

from __future__ import annotations

import re
from typing import Optional

_POLICY_RE = re.compile(r"\bPOL\d{4,8}\b", re.IGNORECASE)
_CUSTOMER_RE = re.compile(r"\bCUST\d{4,8}\b", re.IGNORECASE)
_CLAIM_RE = re.compile(r"\bCLM\d{4,8}\b", re.IGNORECASE)


def extract_policy_number(text: str) -> Optional[str]:
    m = _POLICY_RE.search(text)
    return m.group(0).upper() if m else None


def extract_customer_id(text: str) -> Optional[str]:
    m = _CUSTOMER_RE.search(text)
    return m.group(0).upper() if m else None


def extract_claim_id(text: str) -> Optional[str]:
    m = _CLAIM_RE.search(text)
    return m.group(0).upper() if m else None
