#!/usr/bin/env python3
"""
Fetch from explicit sources (written + adult). Required in every training run.
Written: Literotica, AO3, ASSTR, StoriesOnline.
Adult (titles/descriptions only): Pornhub, Aznude, XHamster.
User has permission for their IP. Saves to data/explicit_scraped.txt.
"""
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output paths — merged by merge_raw_data as compulsory
OUTPUT_PATH = root / "data" / "explicit_scraped.txt"
MEDIA_DIR = root / "data" / "explicit_media"
IMAGES_DIR = MEDIA_DIR / "images"
VIDEOS_DIR = MEDIA_DIR / "videos"
CAPTIONS_PATH = MEDIA_DIR / "captions.json"
CHUNK_SIZE = 800  # Split long text into chunks for training


def _chunk_text(text: str) -> list[str]:
    """Split long text into training-sized chunks."""
    text = " ".join(text.split())
    if len(text) <= CHUNK_SIZE:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            # Prefer splitting at sentence/word boundary
            for sep in ". ", " ", "\n":
                last = text.rfind(sep, start, end + 1)
                if last > start:
                    end = last + len(sep)
                    break
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if len(c) >= 50]


def main():
    """Run all scrapers, save to explicit_scraped.txt."""
    is_kaggle = os.path.exists("/kaggle") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
    allow_local = os.environ.get("ALLOW_LOCAL_CORPUS", "").lower() in ("1", "true", "yes")
    if not is_kaggle and not allow_local:
        logger.info("Skipping scrape (Kaggle only, or set ALLOW_LOCAL_CORPUS=1)")
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text("", encoding="utf-8")
        return 0
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    lines = []

    scrapers = [
        ("Literotica", "scripts.scrapers.literotica", "scrape_literotica"),
        ("AO3", "scripts.scrapers.ao3", "scrape_ao3"),
        ("ASSTR", "scripts.scrapers.asstr", "scrape_asstr"),
        ("StoriesOnline", "scripts.scrapers.storiesonline", "scrape_storiesonline"),
        ("Pornhub", "scripts.scrapers.pornhub", "scrape_pornhub"),
        ("Aznude", "scripts.scrapers.aznude", "scrape_aznude"),
        ("XHamster", "scripts.scrapers.xhamster", "scrape_xhamster"),
    ]

    for name, mod, fn in scrapers:
        try:
            m = __import__(mod, fromlist=[fn])
            scrape_fn = getattr(m, fn)
            count = 0
            for text, source in scrape_fn():
                # Short metadata (adult sites) use as single line; long text chunked
                chunks = _chunk_text(text) if len(text) > CHUNK_SIZE else ([text.strip()] if text and len(text.strip()) >= 50 else [])
                for chunk in chunks:
                    if len(chunk) < 50:
                        continue
                    key = chunk[:200]
                    if key not in seen:
                        seen.add(key)
                        lines.append(chunk)
                        count += 1
            logger.info("Fetched %d chunks from %s", count, name)
        except Exception as e:
            logger.warning("Scraper %s failed: %s", name, e)

    if lines:
        OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Saved %d chunks to %s", len(lines), OUTPUT_PATH)
    else:
        OUTPUT_PATH.write_text("", encoding="utf-8")
        logger.warning("No content scraped; empty file written")

    # Download images and videos from adult scrapers; write captions.json
    media_scrapers = [
        ("Pornhub", "scripts.scrapers.pornhub", "scrape_pornhub_media"),
        ("Aznude", "scripts.scrapers.aznude", "scrape_aznude_media"),
        ("XHamster", "scripts.scrapers.xhamster", "scrape_xhamster_media"),
    ]
    captions_list = []
    from scripts.scrapers.base import download_file, MAX_VIDEO_BYTES

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    for name, mod, fn in media_scrapers:
        try:
            m = __import__(mod, fromlist=[fn])
            scrape_media = getattr(m, fn)
            site_dir_img = IMAGES_DIR / name.lower()
            site_dir_vid = VIDEOS_DIR / name.lower()
            site_dir_img.mkdir(parents=True, exist_ok=True)
            site_dir_vid.mkdir(parents=True, exist_ok=True)
            idx = 0
            for media_url, caption, kind in scrape_media():
                idx += 1
                ext = ".jpg" if kind == "image" else ".mp4"
                safe = hashlib.md5(media_url.encode()).hexdigest()[:12]
                if kind == "image":
                    path = site_dir_img / f"{idx}_{safe}{ext}"
                    if download_file(media_url, path, max_bytes=10 * 1024 * 1024):
                        rel = str(path.relative_to(MEDIA_DIR))
                        captions_list.append({"image_path": rel, "caption": caption or rel})
                elif kind == "video":
                    path = site_dir_vid / f"{idx}_{safe}{ext}"
                    if download_file(media_url, path, max_bytes=MAX_VIDEO_BYTES):
                        captions_list.append({"caption": caption or path.name})
            if idx:
                logger.info("Media from %s: %d items", name, idx)
        except Exception as e:
            logger.warning("Media scraper %s failed: %s", name, e)

    if captions_list:
        CAPTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CAPTIONS_PATH, "w", encoding="utf-8") as f:
            json.dump({"samples": captions_list}, f, ensure_ascii=False, indent=0)
        logger.info("Saved %d media captions to %s", len(captions_list), CAPTIONS_PATH)
    else:
        MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        with open(CAPTIONS_PATH, "w", encoding="utf-8") as f:
            json.dump({"samples": []}, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
