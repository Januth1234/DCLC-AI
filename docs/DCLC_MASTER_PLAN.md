# DCLC Master Plan — Consolidated

Single source of truth for the Dual-Core Latent Composer. All earlier plans merged and updated.

---

## 1. Project Overview

| Aspect | Decision |
|--------|----------|
| **Repo** | [github.com/Januth1234/DCLC-AI](https://github.com/Januth1234/DCLC-AI) |
| **Model size** | 500M parameters |
| **Language** | Sinhala (native, no translations for core) |
| **Training** | Google Colab, 15 GB GPU, ~3 days |
| **Inference** | Dual-core CPU (4+ GB RAM), optional GPU |
| **App** | Windows .exe, Gradio UI |
| **Content** | Unfiltered by default; explicit allowed |

---

## 2. Hardware & Workflow

### Training (Colab)

| Resource | Value |
|----------|-------|
| GPU | 15 GB (T4 or similar) |
| RAM | ~12.7 GB |
| Disk | ~70 GB free (use Drive for data) |
| Time | ~3 days (72 h GPU) |
| Data location | Google Drive (mount in Colab) |

- Use **pretrained VQ** (no VQ training) to fit into 3 days.
- Batch size 1, gradient accumulation 32.
- Save checkpoints every 1500 steps, copy to Drive.

### Inference (User Machines)

| Spec | Min | Recommended |
|------|-----|-------------|
| RAM | 4 GB | 8 GB |
| CPU | Dual-core | Quad-core |
| GPU | None | 4–6 GB VRAM |
| Disk | ~1 GB | ~2 GB |

- 500M @ 4-bit ≈ 250 MB.
- CPU: ~2–4 min per image (128×128).
- GPU: ~30–60 s per image.

---

## 3. Architecture

```
[Text / Image] → Unified Tokenizer (Sinhala BPE + VQ) → 500M Transformer
                                                              ↓
                                                    Latent Grid (16×16)
                                                              ↓
                                                    VQ Decoder → Image
```

- **Unified tokenizer**: Sinhala text BPE (32k) + VQ visual tokens (8192+) in one embedding table.
- **500M model**: 24 layers, 768 hidden, 12 heads.
- **Image edit**: `[IMG] [encoded tokens] [EDIT_START] [instruction]` → modified latent grid → image.

---

## 4. Data Pipeline

### Order of Operations

1. **Download data first** (before Colab).
2. **Train Sinhala tokenizer** (locally or short Colab run).
3. **Upload all data to Drive**.
4. **Run main training** on Colab (3 days).

### Sinhala Text

| Source | Size | Access |
|--------|------|--------|
| Wikipedia Sinhala | ~50 MB | HuggingFace |
| UCSC 10M corpus | ~100 MB | ltrl.ucsc.lk |
| NSina | ~500 MB–2 GB | If available |
| Aggregate | 2–5 GB | Automated script |

- Native Sinhala only for text; no translations for core.
- Script: `aggregate_sinhala_corpus.py` → `data/sinhala/corpus.txt`.

### Image–Text

- Standard datasets (COCO, LAION subset) with Sinhala captions.
- Bootstrap: translate English captions → Sinhala (weak signal); native text dominant.
- Explicit content: path in config (e.g. `data.explicit_path`), gitignored; folder can be hidden.
- Script loads from configured path; path stays out of repo.

### Explicit Content in Training

- Included in training data when path is set.
- Path in **config only** (e.g. `config.local.yaml`); never in repo.
- Folder can be Windows "hidden" (attribute); scripts still read it if path is correct.
- Training script loads from `data.explicit_path` when present; omitted if empty.

### Data Config (gitignored)

```yaml
# config.local.yaml — never committed
data:
  sinhala_path: "data/sinhala"
  image_caption_path: "data/image_captions.json"
  explicit_path: "D:/path/to/hidden/folder"  # optional, user-set; keeps path private
```

---

## 5. Training Phases

### Phase A: Pre-Colab (Local / Short Colab)

1. Run `aggregate_sinhala_corpus.py`.
2. Run `train_sinhala_tokenizer.py`.
3. Download image datasets; translate captions if needed.
4. Upload `data/`, `output/sinhala_tokenizer.json` to Drive.

### Phase B: Colab (3 Days)

1. Mount Drive.
2. Clone repo or copy from Drive.
3. Load pretrained VQ (HuggingFace).
4. Run `train.py --config configs/train_500m_colab.yaml`.
5. Resume from checkpoint on disconnect.

### Config: `train_500m_colab.yaml`

```yaml
model:
  n_layers: 24
  hidden_dim: 768
  n_heads: 12

training:
  batch_size: 1
  gradient_accumulation: 32
  max_steps: 12000
  save_every: 1500
  data_dir: "/content/drive/MyDrive/DCLC_data"
```

---

## 6. Sinhala Tokenizer

- Grapheme-aware pre-tokenization (U+0D80–U+0DFF).
- BPE trained on Sinhala corpus, vocab 32k.
- Special tokens: BOS, EOS, PAD, IMG_START, IMG_END, EDIT_START.
- Files: `grapheme_utils.py`, `sinhala_tokenizer.py`.

---

## 7. Desktop App

- **Framework**: Gradio.
- **Tabs**: Text generation, Image generation, Image edit, Settings.
- **Content toggle (Option C)**: Default OFF = unfiltered; ON = optional filter.
- **Packaging**: PyInstaller → `.exe` (~800 MB–1.2 GB).
- **Fully local**: model, runtime, no internet needed after install.

---

## 8. Content Policy

- **Default**: Unfiltered; explicit content allowed.
- **Toggle**: User can enable filtering in Settings.
- **Training**: No content-based filtering; diverse data.
- **Policy**: `USAGE_POLICY.md` documents user responsibility.

---

## 9. Image Editing

- Input: image + Sinhala instruction.
- VQ encode image → latent tokens.
- Model: `[IMG tokens] [EDIT_START] [instruction]` → modified tokens.
- VQ decode → edited image.
- Optional: inpainting with mask.

---

## 10. GitHub & Sync

- **Repo**: https://github.com/Januth1234/DCLC-AI
- **Author**: nimnaljanuth@gmail.com
- **Sync**: Post-commit hook pushes to `origin main`.
- **Rule**: `.cursor/rules/git-sync.mdc` enforces email and branch.

---

## 11. Implementation Order

| Step | Action |
|------|--------|
| 1 | Project structure, `requirements.txt`, configs |
| 2 | Sinhala tokenizer (grapheme utils, train, save) |
| 3 | Data: download corpus, images, captions |
| 4 | Pretrained VQ integration |
| 5 | Unified tokenizer |
| 6 | 500M transformer model |
| 7 | Training loop |
| 8 | Colab notebook + Drive setup |
| 9 | Train 3 days on Colab |
| 10 | Inference loader + generator |
| 11 | Gradio app + toggle |
| 12 | Image edit support |
| 13 | PyInstaller, .exe build |
| 14 | `USAGE_POLICY.md`, README |

---

## 12. Zero Human Intervention

- Corpus download: automated; skip sources that need forms.
- Tokenizer: automated metrics; proceed if OK.
- Training: single `run_full_pipeline.py` or Colab notebook.
- Path for explicit data: user sets in local config; path gitignored.

---

## 13. Reference Plans

- `dclc_full_architecture_plan` — base architecture
- `dclc_sinhala_native_plan` — Sinhala tokenizer and data
- `dclc_desktop_app_and_toggle` — app and packaging
- `dclc_explicit_content_policy` — content policy
- `dclc_master_implementation_todos` — 500-task checklist
