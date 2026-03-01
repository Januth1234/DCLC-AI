# Raw / uncensored training (18+)

**18+ only.** Explicit training is **compulsory**: every run merges the preloaded default raw corpus (`configs/raw_default.txt`). The model is trained raw; toggling to "good mode" is app-side only (Settings → filter output). Use cloud (Kaggle) for your data.

## Setup

1. Create a **private Kaggle Dataset** with your extra data:
   - Text: `.txt` files or one `corpus_extra.txt`
   - Image captions: `captions.json` — list of `{"caption":"සිංහල විස්තරය"}` or `["cap1","cap2"]` or `{"samples":[{"caption":"..."}]}`

2. Add the dataset to your notebook (right panel → Add input → your dataset).

3. Create `config.local.yaml` in the repo root (gitignored):

```yaml
data:
  sinhala_path: "data/sinhala"
  explicit_path: "/kaggle/input/your-dataset/corpus_extra.txt"
  raw_image_captions_path: "/kaggle/input/your-dataset/captions.json"
```

4. Run the pipeline. `merge_raw_data.py` will merge your data into the corpus before tokenizer training.

## Paths

- `explicit_path`: File or directory of `.txt` — appended to corpus. No content filtering.
- `raw_image_captions_path`: JSON list of `{image_path, caption}` — captions added to corpus.

## Required: scraped explicit sources

Literotica, AO3, ASSTR, and StoriesOnline are **required** — scraped automatically by `fetch_explicit_sources.py`. See [EXPLICIT_SOURCES_REQUIRED.md](EXPLICIT_SOURCES_REQUIRED.md).

## Optional: image/comics (captions)

For image captions from Pixiv, E-Hentai, etc., add via `raw_image_captions_path` in config.local.yaml. See [USER_SOURCES_OPTIONAL.md](USER_SOURCES_OPTIONAL.md).

## Notes

- You supply the data. No built-in downloads for explicit content.
- Model is trained raw; no output filter by default (see USAGE_POLICY.md).
- Low-end: 4GB RAM, dual-core CPU for inference. Use 128 resolution.
