"""Literotica scraper: category → story links → story text."""
import re
import time
from typing import Iterator
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .base import RATE, fetch

# Category indexes for story listings
CATEGORY_URLS = [
    "https://www.literotica.com/c/erotic-fiction",
    "https://www.literotica.com/c/romance",
    "https://www.literotica.com/c/science-fiction",
]
MAX_PAGES = 3  # pages per category
MAX_STORIES = 50  # total stories per run


def _story_links(html: str, base: str) -> list[str]:
    """Extract story URLs from category page."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/s/" in href and "literotica.com" in href:
            links.append(href)
        elif href.startswith("/s/"):
            links.append(urljoin(base, href))
    return list(dict.fromkeys(links))


def _story_text(html: str) -> str:
    """Extract story body. Literotica uses div with story content."""
    soup = BeautifulSoup(html, "html.parser")
    for sel in ["div#content_wrapper", "div.document", "div.story", "article", "div.aa-ht"]:
        el = soup.select_one(sel)
        if el:
            for tag in el.find_all(["script", "style", "nav", "footer"]):
                tag.decompose()
            t = el.get_text(separator=" ", strip=True)
            if len(t) > 100:
                return " ".join(t.split())
    # Fallback: main content area
    main = soup.find("div", class_=re.compile(r"panel|content|story|body"))
    if main:
        return " ".join(main.get_text(separator=" ", strip=True).split())
    return ""


def scrape_literotica(max_stories: int = MAX_STORIES, rate: float = RATE) -> Iterator[tuple[str, str]]:
    """Yield (text, source_url)."""
    seen = set()
    count = 0
    for cat_url in CATEGORY_URLS:
        if count >= max_stories:
            break
        for page in range(1, MAX_PAGES + 1):
            if count >= max_stories:
                break
            url = f"{cat_url.rstrip('/')}/page/{page}" if page > 1 else cat_url
            time.sleep(rate)
            html = fetch(url)
            if not html:
                continue
            links = _story_links(html, url)
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
