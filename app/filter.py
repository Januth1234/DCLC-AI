"""Optional content filter. Bypassed when allow_unfiltered True (default).
Image generation is never filtered; only text may be filtered when Good mode is ON."""
import logging
import re
logger = logging.getLogger(__name__)

# When filter is ON, optionally match keywords (configurable). Empty = no blocking.
# We do not block nudes, crime scenes, or violence by default.
FILTER_KEYWORDS = []


def content_filter_check(allow_unfiltered: bool, prompt: str = "", output_text: str = "") -> bool:
    """Return True if filter should run. When allow_unfiltered True, never filter."""
    if allow_unfiltered:
        return False
    prompt_l = (prompt or "").lower()
    output_l = (output_text or "").lower()
    for kw in FILTER_KEYWORDS:
        if kw.lower() in prompt_l or kw.lower() in output_l:
            return True
    return False


def filter_output(allow_unfiltered: bool, text: str) -> str:
    """If filter enabled and keyword match, optionally redact. Default: pass through."""
    if allow_unfiltered or not FILTER_KEYWORDS:
        return text
    out = text
    for kw in FILTER_KEYWORDS:
        if kw in out:
            out = re.sub(re.escape(kw), "[filtered]", out, flags=re.IGNORECASE)
    return out
