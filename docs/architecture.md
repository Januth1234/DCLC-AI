# DCLC Architecture

- **Model**: Decoder-only transformer, 500M params (24 layers, 768 hidden, 12 heads, 3072 FFN).
- **Tokenizer**: Unified text (Sinhala BPE) + visual (VQ codebook 8192). Text IDs 0..N-1, visual N..N+8191.
- **VQ**: 16×16 latent grid for 256×256 images; straight-through estimator.
- **Training**: Cross-entropy on text and image tokens; AdamW, cosine LR, gradient accumulation.
- **Inference**: Autoregressive text; image = prompt → latent grid → VQ decode. Edit = image + instruction → edited image.
