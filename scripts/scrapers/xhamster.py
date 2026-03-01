"""XHamster scraper: text + image/video download. User has permission."""
import re
import time
from typing import Iterator
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .base import RATE, fetch

LISTING_URLS = [
    "https://xhamster.com/videos",
    "https://xhamster.com/channels",
]
MAX_PAGES = 2
MAX_ITEMS = 40


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
    for sel in ["[class*='title']", "[class*='description']", "[class*='caption']", "a[class*='video']"]:
        for el in soup.select(sel)[:20]:
            txt = el.get_text(separator=" ", strip=True)
            if txt and 10 <= len(txt) <= 2000:
                items.append(txt)
    return items


def _listing_links(html: str, base: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/videos/" in href and "xhamster" in (href if href.startswith("http") else base):
            full = urljoin(base, href)
            if "xhamster.com" in full:
                links.append(full.split("?")[0])
    return list(dict.fromkeys(links))[:30]


def scrape_xhamster(max_items: int = MAX_ITEMS, rate: float = RATE) -> Iterator[tuple[str, str]]:
    """Yield (text, source_url). Text only."""
    seen = set()
    count = 0
    for list_url in LISTING_URLS:
        if count >= max_items:
            break
        for page in range(1, MAX_PAGES + 1):
            if count >= max_items:
                break
            url = f"{list_url}?page={page}" if page > 1 else list_url
            time.sleep(rate)
            html = fetch(url)
            if not html:
                continue
            for text in _extract_text_items(html, url):
                key = text[:100]
                if key not in seen and len(text) >= 15:
                    seen.add(key)
                    count += 1
                    yield (text, url)
            for link in _listing_links(html, url):
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


def _extract_media(html: str, base: str, caption: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    soup = BeautifulSoup(html, "html.parser")
    images = []
    videos = []
    for img in soup.find_all("img", src=True):
        src = img.get("src") or img.get("data-src")
        if src and isinstance(src, str) and not src.startswith("data:"):
            full = urljoin(base, src)
            if "xhamster" in full or "thumb" in full.lower():
                images.append((full, (img.get("alt") or caption or "image")[:200]))
    for s in soup.find_all("source", src=True):
        u = urljoin(base, s["src"])
        if ".mp4" in u or "video" in u:
            videos.append((u, caption[:200]))
    for v in soup.find_all("video"):
        for s in v.find_all("source", src=True):
            videos.append((urljoin(base, s["src"]), caption[:200]))
    return (images[:15], videos[:5])


def scrape_xhamster_media(
    max_images: int = 40,
    max_videos: int = 5,
    rate: float = RATE,
) -> Iterator[tuple[str, str, str]]:
    """Yield (media_url, caption, type)."""
    seen = set()
    img_c, vid_c = 0, 0
    for list_url in LISTING_URLS:
        for page in range(1, MAX_PAGES + 1):
            url = f"{list_url}?page={page}" if page > 1 else list_url
            time.sleep(rate)
            html = fetch(url)
            if not html:
                continue
            soup = BeautifulSoup(html, "html.parser")
            cap = (soup.find("title").get_text(strip=True) if soup.find("title") else "")[:200]
            imgs, vids = _extract_media(html, url, cap)
            for u, c in imgs:
                if u not in seen and img_c < max_images:
                    seen.add(u)
                    img_c += 1
                    yield (u, c, "image")
            for link in _listing_links(html, url)[:8]:
                if vid_c >= max_videos:
                    break
                time.sleep(rate)
                page_html = fetch(link)
                if not page_html:
                    continue
                t = BeautifulSoup(page_html, "html.parser").find("title")
                cap = (t.get_text(strip=True) if t else link)[:200]
                _, vids = _extract_media(page_html, link, cap)
                for u, c in vids:
                    if u not in seen and vid_c < max_videos:
                        seen.add(u)
                        vid_c += 1
                        yield (u, c, "video")
