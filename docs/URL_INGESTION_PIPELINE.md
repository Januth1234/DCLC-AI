# URL Ingestion Pipeline

Fetches URLs, extracts text, filters for Sinhala, outputs `dataset.jsonl` and `dataset.csv`.

## Workflow: Crawl → Ingest

1. **Crawler**: Extracts article links from news site homepages
2. **Ingestion**: Fetches each URL, extracts text, keeps Sinhala only

```bash
# Step 1: Crawl news sites (urls.txt = homepages) → urls_crawled.txt
python scripts/url_crawler.py --input urls.txt --output urls_crawled.txt --include-seeds

# Step 2: Ingest extracted links
python scripts/url_ingestion_pipeline.py --input urls_crawled.txt --output dataset.jsonl
```

## Install

```bash
pip install -r scripts/requirements-ingestion.txt
```

## Crawler (url_crawler.py)

Fetches homepage/category pages, extracts `<a href>`, normalizes to full URLs, deduplicates.

```bash
python scripts/url_crawler.py --input urls.txt --output urls_crawled.txt
python scripts/url_crawler.py --input urls.txt --output urls_crawled.txt --include-seeds  # add homepages too
```

## Ingestion Usage

```bash
# Default: reads urls.txt, writes dataset.jsonl + dataset.csv to current dir
python scripts/url_ingestion_pipeline.py

# Use crawled links
python scripts/url_ingestion_pipeline.py --input urls_crawled.txt --output dataset.jsonl

# Output to directory (writes dataset.jsonl, dataset.csv, failed_urls.txt, summary.txt there)
python scripts/url_ingestion_pipeline.py --input urls.txt --output ./output/

# Adjust rate limiting (seconds between requests)
python scripts/url_ingestion_pipeline.py --input urls.txt --output dataset.jsonl --rate-limit 3

# Headless browser mode (for JavaScript-rendered pages)
pip install playwright && playwright install chromium
python scripts/url_ingestion_pipeline.py --input urls.txt --output dataset.jsonl --headless
```

## Input

`urls.txt` — one URL per line. Supports HTML pages, JSON APIs, and raw text files.

## Output

| File | Description |
|------|-------------|
| `dataset.jsonl` | One JSON object per line: `{"text": "...", "source": "URL"}` |
| `dataset.csv` | Same data in CSV format |
| `failed_urls.txt` | URLs that did not produce a document (fetch fail, non-200, non-Sinhala) |
| `summary.txt` | Total processed, successful, failed, Sinhala kept |

## Kaggle dataset upload

1. Run the pipeline to generate `dataset.jsonl` and `dataset.csv`.

2. Zip the output files:
   ```bash
   zip -r dclc_url_dataset.zip dataset.jsonl dataset.csv
   ```

3. Create a Kaggle Dataset:
   - Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
   - Click **New Dataset**
   - Upload `dclc_url_dataset.zip` or drag the files
   - Add a description and submit

4. In a Kaggle notebook, add the dataset as input. Paths will be:
   ```
   /kaggle/input/your-dataset-name/dataset.jsonl
   /kaggle/input/your-dataset-name/dataset.csv
   ```

## Pipeline behavior

- **Fetching**: User-Agent header, 3 retries with exponential backoff, 2s sleep between requests. Use `--headless` for Playwright/Chromium to handle JavaScript-rendered pages.
- **Content**: HTML → BeautifulSoup (removes script, style, nav, footer); JSON → flattened text; plain text → direct
- **Filtering**: Whitespace normalization, deduplication, min 50 chars, Sinhala-only (`langdetect`)
