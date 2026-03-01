"""Training loop for DCLC."""
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    """Train transformer with cross-entropy."""

    def __init__(self, model, tokenizer, train_loader, config):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 0.01),
        )
        self.grad_accum = config.get("gradient_accumulation", 1)
        self.save_every = config.get("save_every", 2000)
        self.max_steps = config.get("max_steps", 10000)
        self.warmup_steps = config.get("warmup_steps", 0)
        self.base_lr = config.get("lr", 1e-4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.global_step = 0

    def train_step(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids).to(self.device)
        if self.model.training:
            self.model.train()
        logits = self.model.get_logits(self.model(input_ids))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return loss

    def train(self, resume_from: str = None):
        if resume_from and Path(resume_from).exists():
            ckpt = torch.load(resume_from, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt.get("model", ckpt), strict=False)
            self.global_step = ckpt.get("step", 0)
        self.model.train()
        accum_loss = 0.0
        self.optimizer.zero_grad()
        for batch in self.train_loader:
            loss = self.train_step(batch) / self.grad_accum
            loss.backward()
            accum_loss += loss.item()
            if (self.global_step + 1) % self.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get("max_grad_norm", 1.0))
                if self.warmup_steps > 0:
                    step = self.global_step + 1
                    lr = self.base_lr * (step / self.warmup_steps) if step <= self.warmup_steps else self.base_lr
                    for g in self.optimizer.param_groups:
                        g["lr"] = lr
                self.optimizer.step()
                self.optimizer.zero_grad()
                logger.info("Step %d loss %.4f", self.global_step, accum_loss)
                accum_loss = 0.0
            self.global_step += 1
            if self.global_step % self.save_every == 0:
                out = self.config.get("output_dir", "output")
                torch.save({"model": self.model.state_dict(), "step": self.global_step}, f"{out}/checkpoint_{self.global_step}.pt")
            if self.global_step >= self.max_steps:
                break
