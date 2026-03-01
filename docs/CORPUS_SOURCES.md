# Sinhala corpus sources — full reference

DCLC aggregates text from multiple sources. Configured in `configs/sinhala_corpus_sources.yaml`.

---

## Seed sources (trusted starting points)

| Source | Status |
|--------|--------|
| Wikimedia Commons | Dump / manual |
| Wiktionary | Dump / manual |
| Wikisource | Dump / manual |
| Wikiquote | Dump / manual |
| Common Crawl | ✅ CC-100, OSCAR in config |
| Hugging Face | ✅ Primary; multiple datasets |
| Kaggle | Add as input dataset; point via config |
| NSINA | ✅ In config |
| OpenSubtitles | OPUS/HF |
| OPUS | Add OPUS subsets to config |
| LIRNEasia | Policy/research; manual |
| Project Gutenberg | Manual |
| TED Talks | OPUS |
| Ada Derana | Scraping; compliance |
| Hiru News | Scraping; compliance |
| Daily News | Scraping; compliance |
| Lankadeepa | Scraping; compliance |
| LK NLP | lknlp.github.io; manual links |

---

## 📚 Auto-fetched (Hugging Face)

| Source | Dataset | Notes |
|--------|---------|------|
| **Wikimedia Wikipedia** | `wikimedia/wikipedia` (20231101.si) | Sinhala Wikipedia dump |
| **CC-100** | `statmt/cc100` (si) | Large Common Crawl derived (452M chars) |
| **OSCAR** | `oscar-corpus/OSCAR-2201` (si) | Filtered Common Crawl; needs HF token |
| **NSINA** | `sinhala-nlp/NSINA-Categories` | News corpus; accept terms on HF first |
| **Wikipedia 400** | `Minuri/sinhala-corpus-wikipedia400` | Small Wikipedia subset |

---

## 📚 Encyclopedic / Open knowledge

| Source | Access | Notes |
|--------|--------|------|
| **Wikimedia Commons** | dumps.wikimedia.org | Sinhala descriptions, metadata; requires dump processing |
| **Wiktionary (si)** | dumps.wikimedia.org/siwiktionary | Definitions, example sentences, morphology |
| **Wikisource (si)** | dumps.wikimedia.org/siwikisource | Public domain books, historical docs |
| **Wikiquote (si)** | dumps.wikimedia.org/siwikiquote | Dialogue-style structured text |
| **Wikipedia** | ✅ In config | Already used via wikimedia/wikipedia |

---

## 🌍 Large web-crawled corpora

| Source | Access | Notes |
|--------|--------|------|
| **Common Crawl** | ✅ CC-100, OSCAR | CC-100 + OSCAR cover filtered crawls; raw CC needs langid (fastText) |
| **mC4** | `allenai/c4` or `mc4` (si) | Filtered web text; T5-style; add to config if desired |
| **Leipzig Corpora** | wortschatz.uni-leipzig.de | Sinhala may be limited; check for sentence corpora |
| **OPUS** | opus.nlpl.eu / HF | Parallel + monolingual; subtitles, EU, religious; add OPUS subsets to config |

---

## 📰 News & media archives

| Source | Access | Notes |
|--------|--------|------|
| **NSINA** | ✅ In config | 500k+ news articles from major sites |
| **Ada Derana, Hiru, Daily News, Lankadeepa** | Scraping | Requires ToS/compliance checks; add via explicit_path if obtained |
| **LK NLP news** | lknlp.github.io | Curated links; manual download |

---

## 🎬 Subtitles / conversational

| Source | Access | Notes |
|--------|--------|------|
| **OpenSubtitles** | opus.nlpl.eu/OpenSubtitles | Dialogue-rich; informal; HF or OPUS |
| **TED Talks (si subtitles)** | opus.nlpl.eu/TED2020 | Educational spoken style |
| **OPUS subtitles** | HF `opus-100-corpus` or similar | Add Sinhala slice to config |

---

## 🏛 Government & institutional

| Source | Access | Notes |
|--------|--------|------|
| **Gov.lk** | gov.lk | Policy docs, notices; scraping required |
| **Census Dept** | statistics.gov.lk | Reports, structured text |
| **Parliament** | parliament.lk | Proceedings, bills; formal/legal register |
| **Sri Lanka Doc Datasets** | ArXiv / uni repos | 229k+ documents (parliament, law, news) |

Add via `explicit_path` after manual download or compliant scraping.

---

## 📖 Literary & cultural

| Source | Access | Notes |
|--------|--------|------|
| **Project Gutenberg** | gutenberg.org | Limited Sinhala; check manually |
| **Sinhala poetry blogs** | Various | License check required |
| **Folk story archives** | Public archives | Manual curation |

---

## 🧠 Academic / research corpora

Search Google Scholar + university repos (UoM, UCSC, Moratuwa) for:

- Sinhala POS-tagged corpora (e.g. `NLPC-UOM/Sinhala-POS-Data`)
- Morphological datasets
- Sentiment datasets
- NER corpora
- `sinhala-nlp` org on Hugging Face

Many small but clean corpora; add to config or `explicit_path` once obtained.

---

## OSCAR & NSINA access

- **OSCAR**: Accept terms + set `HF_TOKEN` (or `huggingface-cli login`)
- **NSINA**: Accept terms at [sinhala-nlp/NSINA-Categories](https://huggingface.co/datasets/sinhala-nlp/NSINA-Categories)

---

## Adding new sources

1. **Hugging Face**: Add entry to `configs/sinhala_corpus_sources.yaml` with `dataset`, `config`/`language`, `text_column`, optional `max_rows`/`streaming`.
2. **Local / manual**: Download or scrape (legally), save as `.txt` or JSON, point `explicit_path` or `raw_image_captions_path` in `config.local.yaml`.
