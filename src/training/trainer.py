"""Training loop for DCLC. Supports AMP, DDP, gradient checkpointing."""
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _unwrap_model(model):
    return getattr(model, "module", model)


class Trainer:
    """Train transformer with cross-entropy. Optional AMP and DDP (rank 0 saves/logs)."""

    def __init__(self, model, tokenizer, train_loader, config, rank=0):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.config = config
        self.rank = rank
        self.use_amp = config.get("mixed_precision", False) and self._device().type == "cuda"
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 0.01),
        )
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.grad_accum = config.get("gradient_accumulation", 1)
        self.save_every = config.get("save_every", 2000)
        self.max_steps = config.get("max_steps", 10000)
        self.warmup_steps = config.get("warmup_steps", 0)
        self.base_lr = config.get("lr", 1e-4)
        self.device = self._device()
        self.model.to(self.device)
        self.global_step = 0

    def _device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_step(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids).to(self.device)
        if self.model.training:
            self.model.train()
        if self.use_amp:
            with torch.amp.autocast("cuda"):
                logits = self.model.get_logits(self.model(input_ids))
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
                )
            return loss
        logits = self.model.get_logits(self.model(input_ids))
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
        )

    def train(self, resume_from: str = None):
        unwrap = _unwrap_model(self.model)
        if resume_from and Path(resume_from).exists():
            ckpt = torch.load(resume_from, map_location=self.device, weights_only=True)
            unwrap.load_state_dict(ckpt.get("model", ckpt), strict=False)
            self.global_step = ckpt.get("step", 0)
        self.model.train()
        accum_loss = 0.0
        self.optimizer.zero_grad()
        for batch in self.train_loader:
            loss = self.train_step(batch) / self.grad_accum
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += loss.item()
            if (self.global_step + 1) % self.grad_accum == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.get("max_grad_norm", 1.0)
                )
                if self.warmup_steps > 0:
                    step = self.global_step + 1
                    lr = (
                        self.base_lr * (step / self.warmup_steps)
                        if step <= self.warmup_steps
                        else self.base_lr
                    )
                    for g in self.optimizer.param_groups:
                        g["lr"] = lr
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                if self.rank == 0:
                    logger.info("Step %d loss %.4f", self.global_step, accum_loss)
                accum_loss = 0.0
            self.global_step += 1
            if self.global_step % self.save_every == 0 and self.rank == 0:
                out = self.config.get("output_dir", "output")
                Path(out).mkdir(parents=True, exist_ok=True)
                state = unwrap.state_dict()
                torch.save(
                    {"model": state, "step": self.global_step},
                    f"{out}/checkpoint_{self.global_step}.pt",
                )
            if self.global_step >= self.max_steps:
                break
