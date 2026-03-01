"""Pornhub scraper: text + image/video download. User has permission."""
import re
import time
from typing import Iterator
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .base import RATE, fetch

LISTING_URLS = [
    "https://www.pornhub.com/video",
    "https://www.pornhub.com/categories",
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
            if txt and len(txt) > 10 and len(txt) < 2000:
                items.append(txt)
    return items


def _listing_links(html: str, base: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/view_video.php" in href or "/video?" in href:
            full = urljoin(base, href)
            if "pornhub.com" in full:
                links.append(full.split("?")[0] if "?" in full else full)
    return list(dict.fromkeys(links))[:30]


def _extract_media(html: str, base: str, caption: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Extract image and video URLs from page. Returns ([(url, caption)], [(url, caption)])."""
    soup = BeautifulSoup(html, "html.parser")
    images = []
    videos = []
    for img in soup.find_all("img", src=True):
        src = img.get("src") or img.get("data-src") or img.get("data-original")
        if src and ("phncdn" in src or "pornhub" in src or "thumb" in src.lower()):
            full = urljoin(base, src)
            alt = (img.get("alt") or caption or "image")[:200]
            images.append((full, alt))
    for v in soup.find_all("video"):
        for s in v.find_all("source", src=True):
            full = urljoin(base, s["src"])
            videos.append((full, caption[:200]))
    for s in soup.find_all("source", src=True):
        src = s["src"]
        if ".mp4" in src or "video" in src.lower():
            full = urljoin(base, src)
            videos.append((full, caption[:200]))
    return (images[:15], videos[:5])


def scrape_pornhub(max_items: int = MAX_ITEMS, rate: float = RATE) -> Iterator[tuple[str, str]]:
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
            links = _listing_links(html, url)
            for link in links:
                if count >= max_items or link in seen:
                    continue
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


def scrape_pornhub_media(
    max_images: int = 50,
    max_videos: int = 5,
    rate: float = RATE,
) -> Iterator[tuple[str, str, str]]:
    """Yield (media_url, caption, type) for image or video."""
    seen_media = set()
    img_count = 0
    vid_count = 0
    for list_url in LISTING_URLS[:1]:
        time.sleep(rate)
        html = fetch(list_url)
        if not html:
            continue
        caption = ""
        for tag in ["meta[property='og:title']", "title"]:
            el = BeautifulSoup(html, "html.parser").select_one(tag)
            if el and (el.get("content") or el.get_text(strip=True)):
                caption = (el.get("content") or el.get_text(strip=True))[:200]
                break
        imgs, vids = _extract_media(html, list_url, caption)
        for url, cap in imgs:
            if url not in seen_media and img_count < max_images:
                seen_media.add(url)
                img_count += 1
                yield (url, cap, "image")
        links = _listing_links(html, list_url)[:10]
        for link in links:
            if vid_count >= max_videos:
                break
            time.sleep(rate)
            page_html = fetch(link)
            if not page_html:
                continue
            title_el = BeautifulSoup(page_html, "html.parser").find("title")
            cap = (title_el.get_text(strip=True) if title_el else link)[:200]
            _, vids = _extract_media(page_html, link, cap)
            for url, c in vids:
                if url not in seen_media and vid_count < max_videos:
                    seen_media.add(url)
                    vid_count += 1
                    yield (url, c, "video")
