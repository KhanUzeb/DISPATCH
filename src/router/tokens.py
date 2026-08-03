"""Centralized token-count estimation used when providers do not report usage."""

from __future__ import annotations

import re


_TOKENISH_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_input_tokens(text: str) -> int:
    """Estimate tokens without a model tokenizer.

    This deliberately remains an approximation: it counts words and symbols
    and applies a small subword expansion. Keep callers behind this helper so
    a model-specific tokenizer can replace it later without changing APIs.
    """
    tokenish = len(_TOKENISH_RE.findall(text))
    return max(1, (tokenish * 4 + 2) // 3)
