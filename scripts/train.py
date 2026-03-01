"""Training entry point."""
import argparse
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from src.models.dclc_transformer import DCLCTransformer
from src.training.dataset import TextDataset
from src.training.trainer import Trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_500m_colab.yaml")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--resume", default=None)
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    data_dir = train_cfg.get("data_dir", args.data_dir)

    vocab_size = model_cfg.get("vocab_size", 50000)
    model = DCLCTransformer(
        vocab_size=vocab_size,
        hidden_dim=model_cfg.get("hidden_dim", 768),
        n_layers=model_cfg.get("n_layers", 24),
        n_heads=model_cfg.get("n_heads", 12),
        ffn_dim=model_cfg.get("ffn_dim", 3072),
        max_seq_len=model_cfg.get("max_seq_len", 1024),
    )

    corpus_path = Path(data_dir) / "sinhala" / "corpus.txt"
    ds = TextDataset(str(corpus_path))
    if len(ds) == 0:
        ds.lines = ["සිංහල පාඨය"]
    loader = DataLoader(ds, batch_size=train_cfg.get("batch_size", 1), shuffle=True, num_workers=0)

    def collate(batch):
        from src.tokenizers.sinhala_tokenizer import SinhalaTokenizer
        tok = SinhalaTokenizer()
        tok.load()
        ids = [tok.encode(b["text"]) for b in batch]
        max_len = max(len(x) for x in ids)
        pad_id = tok.get_special_token_ids().get("[PAD]", 0)
        padded = [x + [pad_id] * (max_len - len(x)) for x in ids]
        return {"input_ids": torch.tensor(padded, dtype=torch.long), "labels": torch.tensor(padded, dtype=torch.long)}

    loader.collate_fn = collate
    trainer = Trainer(model, None, loader, train_cfg)
    Path("output").mkdir(parents=True, exist_ok=True)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
