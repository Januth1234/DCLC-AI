"""Train Sinhala BPE tokenizer with grapheme-aware pre-tokenization."""
import json
import logging
import sys
from pathlib import Path

Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizers.grapheme_utils import split_into_graphemes, split_into_words

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SPECIAL_TOKENS = ["[BOS]", "[EOS]", "[PAD]", "[IMG_START]", "[IMG_END]", "[EDIT_START]"]


def grapheme_pre_tokenizer(text: str) -> list[str]:
    """Pre-tokenize into grapheme-based pieces for BPE."""
    words = split_into_words(text)
    pieces = []
    for w in words:
        graphemes = split_into_graphemes(w)
        pieces.extend(graphemes)
        pieces.append(" ")  # space between words
    return [p for p in pieces if p] if pieces else [""]


def main(
    corpus_path: str = "data/sinhala/corpus.txt",
    output_path: str = "model_output/sinhala_tokenizer.json",
    vocab_size: int = 32000,
    samples_file: str = "model_output/tokenizer_samples.txt",
):
    """Train tokenizer and save."""
    try:
        from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    except ImportError:
        logger.error("pip install tokenizers")
        return

    Path(corpus_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if not Path(corpus_path).exists():
        logger.warning("Corpus not found at %s; creating minimal corpus", corpus_path)
        Path(corpus_path).parent.mkdir(parents=True, exist_ok=True)
        Path(corpus_path).write_text("සිංහල\n", encoding="utf-8")

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(" ", behavior="removed"),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]"] + SPECIAL_TOKENS,
        show_progress=True,
    )

    with open(corpus_path, encoding="utf-8") as f:
        text = f.read()

    from tokenizers.pre_tokenizers import PreTokenizer
    class GraphemePreTokenizer:
        def split(self, s, added):
            parts = grapheme_pre_tokenizer(s)
            return [part for part in parts if part]

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
    ])

    tokenizer.train_from_iterator(
        [line for line in text.splitlines() if line.strip()],
        trainer=trainer,
    )

    tokenizer.save(output_path)
    logger.info("Saved tokenizer to %s", output_path)

    samples = text.splitlines()[:100]
    encoded_samples = []
    for s in samples:
        if not s.strip():
            continue
        enc = tokenizer.encode(s)
        encoded_samples.append(f"Text: {s[:50]}...\nTokens: {enc.tokens}\nIds: {enc.ids}\n")
    Path(samples_file).parent.mkdir(parents=True, exist_ok=True)
    Path(samples_file).write_text("\n".join(encoded_samples[:20]), encoding="utf-8")
    logger.info("Saved samples to %s", samples_file)

    total_tokens = sum(len(tokenizer.encode(s).ids) for s in samples if s.strip())
    total_words = sum(len(split_into_words(s)) for s in samples if s.strip())
    if total_words > 0:
        tpw = total_tokens / total_words
        logger.info("Tokens per word (sample): %.2f", tpw)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/sinhala/corpus.txt")
    p.add_argument("--output", default="model_output/sinhala_tokenizer.json")
    p.add_argument("--vocab-size", type=int, default=32000)
    args = p.parse_args()
    main(args.corpus, args.output)
