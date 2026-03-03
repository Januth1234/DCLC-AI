"""ASSTR scraper: browse authors/collections → story text."""
import logging
import re
import time
from typing import Iterator
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .base import fetch, RATE

logger = logging.getLogger(__name__)

# ASSTR index and collection pages
INDEX_URL = "https://asstr.org/~Kristen/"
COLLECTION_URLS = ["https://asstr.org/collections/"]
MAX_STORIES = 30


def _extract_links(html: str, base: str) -> list[str]:
    """Extract story/text links from ASSTR page."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".htm") or href.endswith(".html") or "/" in href and not href.startswith("#"):
            full = urljoin(base, href)
            if "asstr.org" in full:
                links.append(full)
    return list(dict.fromkeys(links))


def _story_text(html: str) -> str:
    """ASSTR stories are often in pre or plain div."""
    soup = BeautifulSoup(html, "html.parser")
    pre = soup.find("pre")
    if pre:
        return " ".join(pre.get_text(separator=" ", strip=True).split())
    for tag in soup(["script", "style", "nav"]):
        tag.decompose()
    body = soup.find("body") or soup
    t = body.get_text(separator=" ", strip=True)
    if len(t) > 50:
        return " ".join(t.split())
    return ""


def scrape_asstr(max_stories: int = MAX_STORIES, rate: float = RATE) -> Iterator[tuple[str, str]]:
    """Yield (text, source_url)."""
    seen = set()
    count = 0
    seeds = [INDEX_URL] + COLLECTION_URLS
    logger.info("ASSTR: starting scrape (max_stories=%d, rate=%.2fs)", max_stories, rate)
    for base_url in seeds:
        if count >= max_stories:
            break
        logger.info("ASSTR: fetching index/collection %s", base_url)
        time.sleep(rate)
        html = fetch(base_url)
        if not html:
            logger.info("ASSTR: no HTML for %s, skipping", base_url)
            continue
        links = _extract_links(html, base_url)
        logger.info("ASSTR: found %d links on %s", len(links), base_url)
        for link in links:
            if count >= max_stories or link in seen:
                continue
            seen.add(link)
            time.sleep(rate)
            story_html = fetch(link)
            if not story_html:
                continue
            text = _story_text(story_html)
            if text and len(text) >= 50:
                count += 1
                yield (text, link)
    logger.info("ASSTR: finished with %d stories", count)
