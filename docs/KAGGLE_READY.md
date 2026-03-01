# Kaggle ready — copy-paste code

Enable **GPU (P100)** in Notebook Settings first.

---

## Cell 1 — Clone repo

```python
!git clone https://github.com/Januth1234/DCLC-AI.git
%cd DCLC-AI
```

---

## Cell 2 — Full pipeline (corpus + tokenizer + train)

```python
!python scripts/kaggle_full_run.py
```

---

## Alternative: run step-by-step

If you prefer separate cells:

**Cell A — Clone**
```python
!git clone https://github.com/Januth1234/DCLC-AI.git
%cd DCLC-AI
```

**Cell B — Install**
```python
!pip install -q torch torchvision transformers tokenizers datasets pyyaml tqdm regex accelerate
```

**Cell C — Corpus**
```python
!python scripts/aggregate_sinhala_corpus.py --data-dir data/sinhala
```

**Cell D — Tokenizer**
```python
!python scripts/train_sinhala_tokenizer.py
```

**Cell E — Train**
```python
!python scripts/train.py --config configs/train_500m_colab.yaml --data-dir data
```

---

## Resume after session

To resume from a checkpoint (e.g. after saving to Kaggle Dataset):

```python
!python scripts/train.py --config configs/train_500m_colab.yaml --data-dir data --resume output/checkpoint_1500.pt
```

---

## Save checkpoints

Before session ends:

```python
!zip -r dclc_checkpoints.zip output/
```

Then download `dclc_checkpoints.zip` from the Output panel.
