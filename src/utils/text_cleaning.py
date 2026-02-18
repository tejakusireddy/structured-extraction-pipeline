"""Text cleaning utilities for court opinion processing.

All opinion text passes through these functions before being stored
or sent to an LLM. Uses only stdlib to avoid unnecessary dependencies.
"""

import html
import re
import unicodedata

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINES_RE = re.compile(r"\n{3,}")
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")


def strip_html(text: str) -> str:
    """Remove HTML tags and unescape HTML entities.

    Converts block-level tags (<p>, <br>, <div>, <li>) into newlines
    first so paragraph structure is preserved after tag removal.
    """
    result = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    result = re.sub(r"</(p|div|li|tr|blockquote|h[1-6])>", "\n", result, flags=re.IGNORECASE)
    result = _HTML_TAG_RE.sub("", result)
    return html.unescape(result)


def normalize_unicode(text: str) -> str:
    """Apply NFKC unicode normalization and strip zero-width characters."""
    text = unicodedata.normalize("NFKC", text)
    return _ZERO_WIDTH_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Collapse horizontal whitespace runs and limit consecutive blank lines."""
    text = _MULTI_WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Full cleaning pipeline: HTML → unicode → whitespace."""
    text = strip_html(text)
    text = normalize_unicode(text)
    return normalize_whitespace(text)
