"""Base fetcher: requests, retry, rate limit, media download."""
import time
from pathlib import Path
from typing import Iterator

import requests

UA = "DCLC-Scraper/1.0 (+https://github.com/Januth1234/DCLC-AI)"
RATE = 2.0
RETRIES = 3
# Max size for video download (100MB)
MAX_VIDEO_BYTES = 100 * 1024 * 1024


def fetch(url: str, timeout: int = 30) -> str | None:
    """Fetch URL with retries. Returns HTML/text or None."""
    headers = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,*/*"}
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code != 200:
                return None
            return r.text
        except requests.RequestException:
            if attempt < RETRIES - 1:
                time.sleep(2 ** attempt)
    return None


def download_file(
    url: str,
    path: Path,
    timeout: int = 60,
    max_bytes: int | None = None,
    stream: bool = True,
) -> bool:
    """Download URL to path. If max_bytes set, abort if content-length or stream exceeds. Returns True on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": UA, "Accept": "*/*"}
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, stream=stream)
            if r.status_code != 200:
                return False
            cl = r.headers.get("content-length")
            if cl and max_bytes and int(cl) > max_bytes:
                return False
            with open(path, "wb") as f:
                if stream:
                    written = 0
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                            written += len(chunk)
                            if max_bytes and written > max_bytes:
                                return False
                else:
                    f.write(r.content)
            return path.exists() and path.stat().st_size > 0
        except requests.RequestException:
            if attempt < RETRIES - 1:
                time.sleep(2 ** attempt)
    return False


def extract_text(html: str, selectors: list[tuple[str, str]]) -> str:
    """Try selectors; return first non-empty text. selectors: [(method, expr), ...]."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for method, expr in selectors:
        if method == "css":
            els = soup.select(expr)
        elif method == "id":
            el = soup.find(id=expr)
            els = [el] if el else []
        elif method == "class":
            els = soup.find_all(class_=expr)
        else:
            continue
        for el in els:
            if el:
                t = el.get_text(separator=" ", strip=True)
                if t and len(t) > 50:
                    return " ".join(t.split())
    return ""


def normalize_whitespace(s: str) -> str:
    return " ".join(s.split()) if s else ""
