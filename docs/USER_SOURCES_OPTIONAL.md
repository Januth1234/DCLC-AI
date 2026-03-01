# Optional user-sourced data (image captions)

**Written + adult**: Literotica, AO3, ASSTR, StoriesOnline, Pornhub, Aznude, XHamster are **required** and scraped automatically. See [EXPLICIT_SOURCES_REQUIRED.md](EXPLICIT_SOURCES_REQUIRED.md).

**Image captions** (GlobalComix, Pixiv, etc.) below are optional. If you obtain captions in line with terms and copyright, add via `config.local.yaml` → `raw_image_captions_path`.

---

## Platforms

| Platform | Notes |
|----------|--------|
| **GlobalComix** | Adult section for webcomics. Captions/descriptions you export can go into `captions.json`. |
| **E-Hentai / ExHentai** | Doujinshi/manga archives. Any captions you have (e.g. Sinhala descriptions) → `raw_image_captions_path`. |
| **Pixiv** | R-18 section; artist descriptions. Use only data you’re allowed to use. |
| **Subreddits** (e.g. r/EroticArt) | Curated galleries. Respect Reddit ToS and API terms. |
| **JoyReactor / Rule34** | Image boards; text is often titles/tags. If you compile captions, use `captions.json`. |

**Use in DCLC:** Build a `captions.json` with `[{"caption": "සිංහල විස්තරය"}, ...]` (or `image_path` + `caption`). Point `raw_image_captions_path` at it. Images themselves are not ingested by the current pipeline; only captions are merged into the text corpus.

---

## Adding to training

1. Obtain text or captions in a way that complies with each platform’s terms and copyright.
2. Put files in a **private Kaggle Dataset** (e.g. `corpus_extra.txt`, `captions.json`).
3. In the repo root, create `config.local.yaml`:
   ```yaml
   data:
     explicit_path: "/kaggle/input/your-dataset/corpus_extra.txt"
     raw_image_captions_path: "/kaggle/input/your-dataset/captions.json"
   ```
4. Run the usual pipeline; `merge_raw_data.py` will merge your data into the corpus.

(Required adult/written sources are scraped by the pipeline; see EXPLICIT_SOURCES_REQUIRED.md.)
