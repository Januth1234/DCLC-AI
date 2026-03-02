# LAION as Additional Image/Text Source

LAION is **additive**: your existing data (explicit media, Sinhala corpus) is unchanged. LAION adds another source of image+caption pairs. **Both images and text are used for training.**

---

## What you do (simple steps)

### Option A: One command on Kaggle (easiest)

1. Create a Kaggle notebook, turn on GPU (T4 x2).
2. Clone your repo and run:
   ```
   USE_LAION=1 python scripts/kaggle_train_2b_multimodal_hard.py
   ```
3. That's it. The script will:
   - Fetch LAION (laion/relaion2B-en-research-safe) metadata
   - Download the actual images from URLs
   - Train VQ on your explicit images + LAION images
   - Train the 2B model on Sinhala text + explicit captions + LAION images and captions

### Option B: Do it step by step (local or Kaggle)

1. **Get the list of images and captions** (URLs + text):
   ```
   python scripts/fetch_laion_subset.py --hf-dataset laion/relaion2B-en-research-safe --max-samples 50000
   ```
   This creates `data/laion/filtered.parquet` (a list of image URLs and their captions).

2. **Download the images** (turns URLs into real image files):
   ```
   python scripts/download_laion_images.py --input data/laion/filtered.parquet --output data/laion/webdataset
   ```
   This saves images into `data/laion/webdataset/*.tar`.

3. **Train the model** using your usual pipeline with LAION added:
   ```
   python scripts/train_multimodal.py ... --laion-path data/laion/webdataset --laion-prob 0.5
   ```

---

## Quick start (Kaggle, small subset)

```bash
# With LAION (fetches ~20k samples, downloads images, trains VQ + 2B with LAION mixed in)
USE_LAION=1 python scripts/kaggle_train_2b_multimodal_hard.py
```

Or: `python scripts/kaggle_train_2b_multimodal_hard.py --laion`

## Step-by-step (manual)

### 1. Fetch LAION metadata and filter

**HuggingFace dataset (recommended – uses laion/relaion2B-en-research-safe):**
```bash
python scripts/fetch_laion_subset.py --hf-dataset laion/relaion2B-en-research-safe --max-samples 50000 --out-dir data/laion
```

**Parquet download (alternative):**
```bash
python scripts/fetch_laion_subset.py --subset laion2B-en --num-parts 3 --max-samples 50000 --out-dir data/laion
```

Output: `data/laion/filtered.parquet` (columns: URL, TEXT).

### 2. Download images to webdataset

```bash
python scripts/download_laion_images.py --input data/laion/filtered.parquet --output data/laion/webdataset --image-size 256 --shard-size 1000
```

Output: `data/laion/webdataset/*.tar` (each tar: `{id}.jpg`, `{id}.txt`).

### 3. Train VQ with LAION (optional)

```bash
python scripts/train_visual_vq.py \
  --captions data/explicit_media/captions.json \
  --image-root data/explicit_media \
  --laion-tars data/laion/webdataset \
  --laion-prob 0.5 \
  --max-steps 5000 \
  --epochs 1 \
  --out model_output/visual_vq.pt
```

When using `--laion-tars`, set `--max-steps` (e.g. 5000) because the stream is infinite.

### 4. Train 2B multimodal with LAION

```bash
python scripts/train_multimodal.py \
  --config configs/train_2b_multimodal_hard.yaml \
  --output-dir output_2b_multimodal_hard \
  --vq-checkpoint model_output/visual_vq.pt \
  --laion-path data/laion/webdataset \
  --laion-prob 0.5
```

Or set in `configs/train_2b_multimodal_hard.yaml`:

```yaml
training:
  laion_path: "data/laion/webdataset"
  laion_prob: 0.5   # 50% of image steps from LAION; rest from explicit + text
```

## Config options

| Option | Meaning |
|--------|--------|
| `laion_path` | Path to LAION webdataset tars (folder of `*.tar`). Null = no LAION. |
| `laion_prob` | Fraction of training steps that use LAION image-caption (rest = explicit + Sinhala text). |
| `text_only_prob` | (In MixedModalDataset) Fraction of steps that are text-only. |

## Scale (2B images)

- Use `--num-parts 128` and omit `--max-samples` to target full laion2B-en (~2B rows).
- Download with **distributed img2dataset** (Spark); see [laion5B.md](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion5B.md). Output to S3/GCS.
- Point `laion_path` to the webdataset URL or mounted path; set `laion_prob` high (e.g. 0.9) so most image steps come from LAION.

## Dependencies

Already in `requirements.txt`: `webdataset`, `img2dataset`, `pyarrow`. For fetch script: `pandas`, `pyarrow`, `huggingface-hub`.
