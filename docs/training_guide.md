# Training Guide

1. **Data**: Run `scripts/aggregate_sinhala_corpus.py` then optionally `scripts/translate_captions_to_sinhala.py` for image captions.
2. **Tokenizer**: `scripts/train_sinhala_tokenizer.py` writes `model_output/sinhala_tokenizer.json`. Optionally train VQ with `scripts/train_visual_tokenizer.py`.
3. **Train**: `python scripts/train.py --config configs/train_500m_colab.yaml`. Use `--resume` to continue from checkpoint.
4. **Config**: Copy `configs/config.example.yaml` to `config.local.yaml` and set `data.sinhala_path`, `data.explicit_path` (optional). Colab config: batch 1, grad_accum 32, max_steps 12000.
