# Explicit sources — all required (18+)

**18+ only.** Explicit content is **required** in every training run. The pipeline scrapes from all of the following (user has permission for their IP):

## Written (text)

| Source | Description |
|--------|-------------|
| **Literotica** | Adult fiction, romance, sci‑fi |
| **AO3** | Archive of Our Own — Explicit-rated fanfiction |
| **ASSTR** | Alt.Sex.Stories Text Repository |
| **StoriesOnline** | Long-form adult fiction |

## Adult (text + image/video download)

| Source | Description |
|--------|-------------|
| **Pornhub** | Titles, descriptions, tags; **images and videos downloaded** to `data/explicit_media/` |
| **Aznude** | Titles, captions; **images downloaded** |
| **XHamster** | Titles, descriptions; **images and videos downloaded** |

Images go to `data/explicit_media/images/{site}/`, videos to `data/explicit_media/videos/{site}/`. Captions are written to `data/explicit_media/captions.json` and merged into the corpus. Video downloads are capped (size/count) to avoid disk overflow.

## Pipeline flow

1. `fetch_explicit_sources.py` — scrapes all seven sources, saves to `data/explicit_scraped.txt`
2. `merge_raw_data.py` — merges scraped content into the corpus (compulsory)

## Where it runs

- **Kaggle**: Runs automatically.
- **Local / server**: Set `ALLOW_LOCAL_CORPUS=1` when you have permission for your IP.

## Optional: extra image captions

For additional image captions (GlobalComix, Pixiv, E-Hentai, etc.), add via `config.local.yaml` → `raw_image_captions_path`. See [RAW_TRAINING.md](RAW_TRAINING.md).
