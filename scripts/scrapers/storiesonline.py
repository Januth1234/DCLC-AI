"""StoriesOnline scraper: list → story links → story text."""
import re
import time
from typing import Iterator
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .base import fetch, RATE

LIST_URL = "https://storiesonline.net/list/"
MAX_STORIES = 30


def _story_links(html: str, base: str) -> list[str]:
    """Extract story URLs from listing."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/s/" in href or "story" in href.lower():
            full = urljoin(base, href)
            if "storiesonline.net" in full:
                links.append(full.split("?")[0])
    return list(dict.fromkeys(links))


def _story_text(html: str) -> str:
    """Extract story content."""
    soup = BeautifulSoup(html, "html.parser")
    for sel in ["div.story-body", "div.storyContent", "div.content", "article", "div#story"]:
        el = soup.select_one(sel)
        if el:
            for tag in el.find_all(["script", "style", "nav"]):
                tag.decompose()
            t = el.get_text(separator=" ", strip=True)
            if len(t) > 100:
                return " ".join(t.split())
    main = soup.find("div", class_=re.compile(r"story|content|body"))
    if main:
        return " ".join(main.get_text(separator=" ", strip=True).split())
    return ""


def scrape_storiesonline(max_stories: int = MAX_STORIES, rate: float = RATE) -> Iterator[tuple[str, str]]:
    """Yield (text, source_url)."""
    seen = set()
    count = 0
    time.sleep(rate)
    html = fetch(LIST_URL)
    if not html:
        return
    links = _story_links(html, LIST_URL)
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
