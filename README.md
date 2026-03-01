# DCLC-AI

Dual-Core Latent Composer — unified Sinhala text and image model (500M params). No diffusion; latent VQ.

- **Master plan**: [docs/DCLC_MASTER_PLAN.md](docs/DCLC_MASTER_PLAN.md)
- **Usage policy**: [USAGE_POLICY.md](USAGE_POLICY.md) — no content restrictions by default; optional filter in Settings.

## Installation

- Python 3.10+
- Create a virtual environment (recommended):

  ```bash
  python -m venv .venv
  .venv\Scripts\activate   # Windows
  ```

- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

- For building the app executable: `pip install -r requirements-app.txt`

## Run

- **Gradio app** (text / image / edit):

  ```bash
  python -m app.main
  ```

- **Training** (after corpus and tokenizer are ready):

  ```bash
  python scripts/train.py --config configs/train_500m_colab.yaml
  ```

- **Full pipeline** (corpus → tokenizer → optional VQ → train):

  ```bash
  python scripts/run_full_pipeline.py --data-dir data
  ```

## Hardware

- **Inference**: Dual-core CPU, 4GB+ RAM (no GPU required).
- **Training**: Colab 15GB GPU recommended (~3 days for 500M).

## Screenshots

*(Placeholder: add app screenshots here if desired.)*

## Config

- App config: `output/app_config.json` (content filter, resolution, model path, seed).
- Training: copy `configs/config.example.yaml` to `config.local.yaml`; set `data.sinhala_path`, optional `data.explicit_path`.

## FAQ / Troubleshooting

- **No GPU**: DCLC runs on CPU (dual-core, 4GB+ RAM). Training is on Colab or GPU.
- **Tokenizer not found**: Run the full pipeline or `scripts/train_sinhala_tokenizer.py` after aggregating corpus.
- **Out of memory**: Use smaller batch size, enable gradient checkpointing, or use 128 resolution in the app.
- **GPU Code 43**: See `scripts/FIX_GPU_CODE_43.md` if you encounter Windows GPU errors.

## License

See [LICENSE](LICENSE). MIT.