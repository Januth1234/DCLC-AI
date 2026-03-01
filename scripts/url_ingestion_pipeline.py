#!/usr/bin/env python3
"""
URL ingestion pipeline for Sinhala text extraction.
Fetches URLs, extracts text, filters by language, outputs dataset.jsonl + dataset.csv.
"""
import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

# Optional dependencies - fail gracefully with helpful message
try:
    import requests
except ImportError:
    print("Install dependencies: pip install -r scripts/requirements-ingestion.txt")
    sys.exit(1)

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Install beautifulsoup4: pip install beautifulsoup4")
    sys.exit(1)

try:
    from langdetect import detect, LangDetectException
except ImportError:
    print("Install langdetect: pip install langdetect")
    sys.exit(1)

# Optional: headless browser (Playwright) for JS-rendered pages
_playwright_available = False
try:
    from playwright.sync_api import sync_playwright
    _playwright_available = True
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Default User-Agent (polite crawler)
DEFAULT_UA = "DCLC-URL-Ingestion/1.0 (+https://github.com/Januth1234/DCLC-AI)"
MIN_TEXT_LEN = 50
SINHALA_LANG = "si"
RATE_LIMIT_SLEEP = 2.0  # seconds between requests
MAX_RETRIES = 3


def load_urls(path: str) -> list[str]:
    """Load URLs from text file, one per line."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    lines = p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    return [u.strip() for u in lines if u.strip() and not u.strip().startswith("#")]


def fetch_url(url: str, headers: dict, timeout: int = 30) -> requests.Response | None:
    """
    Fetch URL with retries and exponential backoff.
    Returns Response if 200, None otherwise (caller handles non-200).
    """
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            return r
        except requests.RequestException as e:
            wait = 2 ** attempt
            logger.warning("Attempt %d/%d failed for %s: %s. Retry in %ds", attempt + 1, MAX_RETRIES, url, e, wait)
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
    return None


def fetch_with_browser(url: str, timeout: int = 30000) -> tuple[str, int] | None:
    """
    Fetch URL using headless Chromium (Playwright).
    Returns (html_content, status_code) or None on failure.
    Handles JavaScript-rendered pages.
    """
    if not _playwright_available:
        logger.warning("Playwright not installed. Run: pip install playwright && playwright install chromium")
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=DEFAULT_UA,
                viewport={"width": 1280, "height": 720},
            )
            page = context.new_page()
            try:
                resp = page.goto(url, wait_until="networkidle", timeout=timeout)
                status = resp.status if resp else 0
                if status != 200:
                    browser.close()
                    return None
                # Wait for body to render (helps with some SPAs)
                page.wait_for_load_state("domcontentloaded")
                html = page.content()
                browser.close()
                return (html, status)
            except Exception as e:
                logger.warning("Browser fetch failed for %s: %s", url, e)
                browser.close()
                return None
    except Exception as e:
        logger.warning("Playwright failed for %s: %s", url, e)
        return None


def extract_html_text(html: str) -> str:
    """Extract visible text from HTML; remove script, style, nav, footer."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return normalize_whitespace(text)


def flatten_json_to_text(obj, parts: list[str] | None = None) -> str:
    """Recursively flatten JSON values into text."""
    if parts is None:
        parts = []
    if isinstance(obj, dict):
        for v in obj.values():
            flatten_json_to_text(v, parts)
    elif isinstance(obj, list):
        for v in obj:
            flatten_json_to_text(v, parts)
    elif isinstance(obj, str) and obj.strip():
        parts.append(obj.strip())
    elif isinstance(obj, (int, float)):
        parts.append(str(obj))
    elif obj is True or obj is False:
        parts.append(str(obj))
    return " ".join(parts)


def extract_json_text(content: str) -> str:
    """Parse JSON and convert to flattened text."""
    try:
        data = json.loads(content)
        text = flatten_json_to_text(data)
        return normalize_whitespace(text)
    except json.JSONDecodeError:
        return ""


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace to single spaces, strip."""
    if not text:
        return ""
    return " ".join(text.split())


def process_content(url: str, raw: str, content_type: str = "") -> str | None:
    """Extract text from raw content. Returns None if not extractable."""
    ct = (content_type or "").lower()

    if "application/json" in ct or url.endswith(".json"):
        text = extract_json_text(raw)
    elif "text/html" in ct or "text/xhtml" in ct or not ct:
        text = extract_html_text(raw)
    elif "text/plain" in ct or url.endswith(".txt"):
        text = normalize_whitespace(raw)
    else:
        text = normalize_whitespace(raw)

    return text if text else None


def is_sinhala(text: str) -> bool:
    """Detect if text is Sinhala. Returns False on detect failure or short text."""
    if len(text.strip()) < MIN_TEXT_LEN:
        return False
    try:
        lang = detect(text)
        return lang == SINHALA_LANG
    except LangDetectException:
        return False


def run_pipeline(
    input_path: str,
    output_path: str,
    rate_limit: float = RATE_LIMIT_SLEEP,
    use_headless: bool = False,
) -> dict:
    """
    Run the full ingestion pipeline.
    Returns summary dict: total_urls, successful, failed, sinhala_kept.
    """
    urls = load_urls(input_path)
    total = len(urls)
    successful = 0  # Fetched 200 and extracted content
    failed = []    # Any URL that did not produce a document (logged to failed_urls.txt)
    seen_texts = set()
    documents = []

    out = Path(output_path)
    if out.suffix == ".jsonl":
        out_dir = out.parent if str(out.parent) else Path(".")
        jsonl_path = out.resolve()
        csv_path = out_dir / (out.stem + ".csv")
    else:
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_dir / "dataset.jsonl"
        csv_path = out_dir / "dataset.csv"

    out_dir = out_dir.resolve()
    failed_path = out_dir / "failed_urls.txt"
    summary_path = out_dir / "summary.txt"

    headers = {"User-Agent": DEFAULT_UA, "Accept": "text/html,application/json,text/plain,*/*"}

    for i, url in enumerate(urls):
        if rate_limit > 0 and i > 0:
            time.sleep(rate_limit)

        raw, content_type = None, ""
        if use_headless and _playwright_available:
            result = fetch_with_browser(url)
            if result:
                raw, _ = result
                content_type = "text/html"
        if raw is None:
            resp = fetch_url(url, headers)
            if resp is None:
                failed.append(url)
                continue
            if resp.status_code != 200:
                failed.append(url)
                continue
            raw = resp.text
            content_type = resp.headers.get("content-type", "")

        text = process_content(url, raw, content_type)
        if not text or len(text) < MIN_TEXT_LEN:
            failed.append(url)
            continue

        # Deduplicate by text fingerprint
        key = text[:500]
        if key in seen_texts:
            successful += 1
            continue  # Don't add duplicate; don't count as failed
        seen_texts.add(key)

        if not is_sinhala(text):
            failed.append(url)
            continue

        successful += 1
        documents.append({"text": text, "source": url})

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "source"])
        w.writeheader()
        w.writerows(documents)

    with open(failed_path, "w", encoding="utf-8") as f:
        f.write("\n".join(failed))

    summary = {
        "total_urls": total,
        "successful": successful,
        "failed": len(failed),
        "sinhala_kept": len(documents),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Total URLs processed: {summary['total_urls']}\n")
        f.write(f"Total successful: {summary['successful']}\n")
        f.write(f"Total failed: {summary['failed']}\n")
        f.write(f"Total Sinhala documents kept: {summary['sinhala_kept']}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="URL ingestion pipeline for Sinhala text")
    parser.add_argument("--input", "-i", default="urls.txt", help="Input file with one URL per line")
    parser.add_argument("--output", "-o", default="dataset.jsonl", help="Output path (dir or .jsonl file)")
    parser.add_argument("--rate-limit", type=float, default=RATE_LIMIT_SLEEP, help="Sleep between requests (seconds)")
    parser.add_argument("--headless", action="store_true", help="Use headless Chromium (Playwright) for JS-rendered pages")
    args = parser.parse_args()

    if args.headless and not _playwright_available:
        logger.error("--headless requires: pip install playwright && playwright install chromium")
        return 1

    try:
        summary = run_pipeline(args.input, args.output, rate_limit=args.rate_limit, use_headless=args.headless)
        logger.info("Done. Sinhala documents kept: %d", summary["sinhala_kept"])
        return 0
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
