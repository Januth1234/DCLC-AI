# Performance tuning

- **CPU**: Use fewer layers or smaller batch for faster inference. `torch.compile(model)` (PyTorch 2+) can speed up.
- **Memory**: Reduce `max_seq_len`, use 128 resolution, enable gradient checkpointing when training. For 4GB devices, close other apps.
- **Training**: Mixed precision (`torch.autocast`), gradient accumulation, and `num_workers` in DataLoader can help.
