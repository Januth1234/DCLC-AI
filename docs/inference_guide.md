# Inference Guide

- **App**: `python -m app.main` (optional: `--port`, `--unhinged`, `--offline`).
- **Model**: Place checkpoint in `output/` or set `model_path` in app config (`output/app_config.json`).
- **Text**: Encode prompt with Sinhala tokenizer; autoregressive decode until EOS or max_length.
- **Image**: Encode prompt → model generates latent token grid → decode with VQ decoder to 256×256 or 128×128.
- **Edit**: Input image + instruction; model predicts edited latent sequence; decode to image.
