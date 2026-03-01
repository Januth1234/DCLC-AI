"""Aznude scraper: text + image download. User has permission."""
import re
import time
from typing import Iterator
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .base import RATE, fetch

LISTING_URLS = [
    "https://www.aznude.com/",
    "https://www.aznude.com/celebrities",
]
MAX_PAGES = 2
MAX_ITEMS = 30


def _extract_text_items(html: str, base: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    items = []
    for tag in soup.find_all("meta", attrs={"name": re.compile(r"description|keywords", re.I)}):
        if tag.get("content"):
            items.append(tag["content"].strip())
    for tag in soup.find_all("meta", attrs={"property": re.compile(r"og:title|og:description", re.I)}):
        if tag.get("content"):
            items.append(tag["content"].strip())
    t = soup.find("title")
    if t and t.get_text(strip=True):
        items.append(t.get_text(strip=True))
    for el in soup.find_all(["h1", "h2", "h3"]):
        txt = el.get_text(strip=True)
        if txt and len(txt) > 5:
            items.append(txt)
    for el in soup.find_all("img", alt=True):
        if el["alt"] and len(el["alt"]) > 10:
            items.append(el["alt"].strip())
    for sel in ["[class*='title']", "[class*='caption']", "[class*='description']"]:
        for el in soup.select(sel)[:15]:
            txt = el.get_text(separator=" ", strip=True)
            if txt and 15 <= len(txt) <= 1500:
                items.append(txt)
    return items


def _page_links(html: str, base: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "aznude.com" in href or href.startswith("/"):
            full = urljoin(base, href)
            if "aznude.com" in full and full != base:
                links.append(full.split("?")[0])
    return list(dict.fromkeys(links))[:25]


def scrape_aznude(max_items: int = MAX_ITEMS, rate: float = RATE) -> Iterator[tuple[str, str]]:
    """Yield (text, source_url)."""
    seen = set()
    count = 0
    for list_url in LISTING_URLS:
        if count >= max_items:
            break
        time.sleep(rate)
        html = fetch(list_url)
        if not html:
            continue
        for text in _extract_text_items(html, list_url):
            key = text[:100]
            if key not in seen and len(text) >= 15:
                seen.add(key)
                count += 1
                yield (text, list_url)
        for link in _page_links(html, list_url):
            if count >= max_items:
                break
            time.sleep(rate)
            page_html = fetch(link)
            if not page_html:
                continue
            for text in _extract_text_items(page_html, link):
                key = text[:100]
                if key not in seen and len(text) >= 15:
                    seen.add(key)
                    count += 1
                    yield (text, link)


def _extract_images(html: str, base: str, caption: str) -> list[tuple[str, str]]:
    out = []
    soup = BeautifulSoup(html, "html.parser")
    for img in soup.find_all("img", src=True):
        src = img.get("src") or img.get("data-src") or img.get("data-original")
        if src and not src.startswith("data:"):
            full = urljoin(base, src)
            alt = (img.get("alt") or caption or "image")[:200]
            out.append((full, alt))
    return out[:20]


def scrape_aznude_media(max_images: int = 60, rate: float = RATE) -> Iterator[tuple[str, str, str]]:
    """Yield (media_url, caption, 'image')."""
    seen = set()
    count = 0
    for list_url in LISTING_URLS:
        time.sleep(rate)
        html = fetch(list_url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        title_el = soup.find("title")
        caption = (title_el.get_text(strip=True) if title_el else "")[:200]
        for url, cap in _extract_images(html, list_url, caption):
            if url not in seen and count < max_images:
                seen.add(url)
                count += 1
                yield (url, cap, "image")
        for link in _page_links(html, list_url)[:15]:
            if count >= max_images:
                break
            time.sleep(rate)
            page_html = fetch(link)
            if not page_html:
                continue
            t = BeautifulSoup(page_html, "html.parser").find("title")
            cap = (t.get_text(strip=True) if t else link)[:200]
            for url, c in _extract_images(page_html, link, cap):
                if url not in seen and count < max_images:
                    seen.add(url)
                    count += 1
                    yield (url, c, "image")
