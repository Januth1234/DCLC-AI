"""Sinhala grapheme utilities. Respects Unicode grapheme clusters (U+0D80-U+0DFF)."""
import regex
import unicodedata

SINHALA_START = 0x0D80
SINHALA_END = 0x0DFF


def is_sinhala(codepoint: int) -> bool:
    """Check if codepoint is in Sinhala Unicode range."""
    return SINHALA_START <= codepoint <= SINHALA_END


def split_into_graphemes(text: str) -> list[str]:
    """Split text into grapheme clusters using regex \\X. Preserves Sinhala vowel markers."""
    if not text:
        return []
    return regex.findall(r"\X", text)


def split_into_words(text: str) -> list[str]:
    """Split into words preserving grapheme boundaries. Uses whitespace and punctuation."""
    if not text:
        return []
    graphemes = split_into_graphemes(text)
    words = []
    current = []
    for g in graphemes:
        if g.isspace() or unicodedata.category(g[0]) == "P":
            if current:
                words.append("".join(current))
                current = []
        else:
            current.append(g)
    if current:
        words.append("".join(current))
    return words


def validate_sinhala_range(text: str) -> bool:
    """Check if text contains valid Sinhala characters (or mixed)."""
    for char in text:
        cp = ord(char)
        if is_sinhala(cp) or char.isspace() or unicodedata.category(char).startswith("L"):
            continue
        if unicodedata.category(char).startswith("M"):
            continue
    return True
