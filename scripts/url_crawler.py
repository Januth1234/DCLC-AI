#!/usr/bin/env python3
"""
Crawl news sites: fetch homepage/category pages, extract article links,
normalize to full URLs, deduplicate. Output to urls.txt for the ingestion pipeline.
"""
import argparse
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

try:
    import requests
except ImportError:
    print("Install: pip install requests")
    sys.exit(1)
try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Install: pip install beautifulsoup4")
    sys.exit(1)

DEFAULT_UA = "DCLC-URL-Crawler/1.0 (+https://github.com/Januth1234/DCLC-AI)"
RATE_LIMIT = 2.0
MAX_RETRIES = 3


def load_seed_urls(path: str) -> list[str]:
    """Load seed URLs from file, one per line."""
    p = Path(path)
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    return [u.strip() for u in lines if u.strip() and not u.strip().startswith("#")]


def fetch_page(url: str, timeout: int = 30) -> str | None:
    """Fetch page HTML. Returns None on failure."""
    headers = {"User-Agent": DEFAULT_UA, "Accept": "text/html,*/*"}
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code != 200:
                return None
            return r.text
        except requests.RequestException:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return None


def same_site(base_netloc: str, link_netloc: str) -> bool:
    """True if link is same site (exact or subdomain)."""
    if link_netloc == base_netloc:
        return True
    if link_netloc.endswith("." + base_netloc):
        return True
    # e.g. seed www.adaderana.lk, link news.adaderana.lk
    parts = base_netloc.split(".")
    if len(parts) >= 2:
        base_domain = ".".join(parts[-2:])  # adaderana.lk
        link_parts = link_netloc.split(".")
        if len(link_parts) >= 2:
            link_domain = ".".join(link_parts[-2:])
            if link_domain == base_domain:
                return True
    return False


def extract_links(html: str, base_url: str) -> list[str]:
    """Extract <a href="..."> links, normalize to absolute URLs, same-site only."""
    soup = BeautifulSoup(html, "html.parser")
    base_netloc = urlparse(base_url).netloc
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        if not parsed.scheme or parsed.scheme not in ("http", "https"):
            continue
        if not same_site(base_netloc, parsed.netloc):
            continue
        links.append(full)
    return links


def normalize_url(url: str) -> str:
    """Remove fragment, trailing slash for consistency."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def crawl_seeds(seeds: list[str], rate_limit: float = RATE_LIMIT) -> set[str]:
    """Crawl seed URLs, extract article links, deduplicate."""
    all_links = set()
    for i, url in enumerate(seeds):
        if rate_limit > 0 and i > 0:
            time.sleep(rate_limit)
        html = fetch_page(url)
        if not html:
            continue
        links = extract_links(html, url)
        for link in links:
            all_links.add(normalize_url(link))
    return all_links


def main():
    parser = argparse.ArgumentParser(description="Crawl news sites, extract article links")
    parser.add_argument("--input", "-i", default="urls.txt", help="Seed URLs (homepages / category pages)")
    parser.add_argument("--output", "-o", default="urls_crawled.txt", help="Output file for extracted links")
    parser.add_argument("--rate-limit", type=float, default=RATE_LIMIT, help="Sleep between requests (seconds)")
    parser.add_argument("--append", action="store_true", help="Append to output instead of overwrite")
    parser.add_argument("--include-seeds", action="store_true", help="Include seed URLs in output (homepages + articles)")
    args = parser.parse_args()

    seeds = load_seed_urls(args.input)
    if not seeds:
        print("No seed URLs found. Add URLs to", args.input)
        return 1

    print(f"Crawling {len(seeds)} seed URLs...")
    links = crawl_seeds(seeds, rate_limit=args.rate_limit)
    if args.include_seeds:
        for s in seeds:
            links.add(normalize_url(s))
    print(f"Extracted {len(links)} unique links")

    out = Path(args.output)
    existing = set()
    if args.append and out.exists():
        existing = set(out.read_text(encoding="utf-8").splitlines())

    to_write = sorted(links | existing)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(to_write) + "\n", encoding="utf-8")
    print(f"Saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
