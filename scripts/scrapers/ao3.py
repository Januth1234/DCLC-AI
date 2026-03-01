"""AO3 scraper: search Explicit works → story text."""
import re
import time
from typing import Iterator

from bs4 import BeautifulSoup

from .base import fetch, RATE

# AO3 search for Explicit-rated works (various fandoms)
SEARCH_URL = "https://archiveofourown.org/works?work_search%5Brating_ids%5D=13"
MAX_PAGES = 5
MAX_STORIES = 50


def _work_links(html: str, base: str) -> list[str]:
    """Extract work URLs from search/listing page."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/works/" in href and "?" not in href.split("/works/")[-1]:
            full = href if href.startswith("http") else f"https://archiveofourown.org{href}"
            links.append(full.split("?")[0])
    return list(dict.fromkeys(links))


def _work_text(html: str) -> str:
    """Extract work content. AO3 uses div#chapters or div.work."""
    soup = BeautifulSoup(html, "html.parser")
    for sel in ["div#chapters", "div.userstuff", "div.work meta", "section.chapters"]:
        el = soup.select_one(sel)
        if el and sel != "div.work meta":
            for tag in el.find_all(["script", "style", "nav"]):
                tag.decompose()
            t = el.get_text(separator=" ", strip=True)
            if len(t) > 100:
                return " ".join(t.split())
    pre = soup.find("pre", class_=re.compile(r"userstuff|works|chapters"))
    if pre:
        return " ".join(pre.get_text(separator=" ", strip=True).split())
    return ""


def scrape_ao3(max_stories: int = MAX_STORIES, rate: float = RATE) -> Iterator[tuple[str, str]]:
    """Yield (text, source_url)."""
    seen = set()
    count = 0
    for page in range(1, MAX_PAGES + 1):
        if count >= max_stories:
            break
        url = f"{SEARCH_URL}&page={page}" if page > 1 else SEARCH_URL
        time.sleep(rate)
        html = fetch(url)
        if not html:
            continue
        links = _work_links(html, url)
        for link in links:
            if count >= max_stories or link in seen:
                continue
            seen.add(link)
            time.sleep(rate)
            work_html = fetch(link)
            if not work_html:
                continue
            text = _work_text(work_html)
            if text and len(text) >= 50:
                count += 1
                yield (text, link)
