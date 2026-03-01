# DCLC-AI

**Dual-Core Latent Composer** — unified Sinhala text and image model (500M / 1B params). No diffusion; latent VQ.

---

## ⚠️ 18+ Only — Consent

**This project is 18+ only.** All content, data sources, and model outputs may include explicit adult material. By **downloading, using, or interacting with DCLC in any way**, you confirm that:

- You are **18 years of age or older** (or the age of majority in your jurisdiction).
- You **consent** to exposure to adult and explicit content.
- You use the model at your own responsibility and in compliance with local laws.

The authors do not restrict content; user responsibility and consent are assumed.

---

## Quick links

- **Usage policy**: [USAGE_POLICY.md](USAGE_POLICY.md) — 18+, consent, no content restrictions by default.
- **Master plan**: [docs/DCLC_MASTER_PLAN.md](docs/DCLC_MASTER_PLAN.md)
- **Kaggle training**: [docs/KAGGLE_READY.md](docs/KAGGLE_READY.md) — 2 cells to run full pipeline.
- **Raw / explicit training**: [docs/RAW_TRAINING.md](docs/RAW_TRAINING.md) · [docs/EXPLICIT_SOURCES_REQUIRED.md](docs/EXPLICIT_SOURCES_REQUIRED.md)

---

## Installation (18+)

- Python 3.10+
- Create a virtual environment (recommended):

  ```bash
  python -m venv .venv
  .venv\Scripts\activate   # Windows
  source .venv/bin/activate   # Linux/macOS
  ```

- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

- For the Gradio app: `pip install -r requirements-app.txt` (if present).

---

## Run

- **Gradio app** (text / image / edit / annotate; 18+):

  ```bash
  python -m app.main
  ```

- **Training** is intended for **Kaggle** (GPU). See [docs/KAGGLE_READY.md](docs/KAGGLE_READY.md):

  **Cell 1:**
  ```python
  !git clone https://github.com/Januth1234/DCLC-AI.git
  %cd DCLC-AI
  ```

  **Cell 2:**
  ```python
  !python scripts/kaggle_full_run.py
  ```

  Pipeline: install deps → fetch explicit sources (written + adult) → aggregate Sinhala corpus → merge raw data → train tokenizer → train 500M model. Checkpoints in `output/`.

---

## Hardware

- **Inference**: Dual-core CPU, 4GB+ RAM (low-end). No GPU required.
- **Training**: Kaggle P100 GPU recommended (~3 days for 500M).

---

## Data and sources (18+)

- **Sinhala corpus**: Wikimedia Wikipedia, CC-100, OSCAR, NSINA, mC4 (see [docs/CORPUS_SOURCES.md](docs/CORPUS_SOURCES.md)).
- **Explicit (required)**: Literotica, AO3, ASSTR, StoriesOnline; Pornhub, Aznude, XHamster (text + image/video download). See [docs/EXPLICIT_SOURCES_REQUIRED.md](docs/EXPLICIT_SOURCES_REQUIRED.md).

---

## Config

- **App**: `output/app_config.json` (content filter toggle, resolution, model path).
- **Training**: `config.local.yaml` (optional extra paths). See `configs/config.example.yaml`.

---

## License

[MIT](LICENSE). Use implies you are 18+ and consent to adult content.
