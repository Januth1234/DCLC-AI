"""Text and image generation."""
import torch
from src.models.dclc_transformer import DCLCTransformer


def generate_text(
    model: DCLCTransformer,
    tokenizer,
    prompt: str,
    max_length: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
) -> str:
    """Autoregressive text generation."""
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_length - len(ids)):
        with torch.no_grad():
            logits = model.get_logits(model(input_ids))[:, -1]
        if temperature > 0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            if top_k > 0:
                v, _ = torch.topk(probs, top_k)
                probs = torch.where(probs < v[:, -1:], torch.zeros_like(probs), probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            next_id = torch.multinomial(probs, 1)
        else:
            next_id = logits.argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        if next_id.item() == tokenizer.get_special_token_ids().get("[EOS]", -1):
            break
    return tokenizer.decode(input_ids[0].tolist())


def generate_image(model, tokenizer, vq_decoder, prompt: str, grid_size: int = 16, **kwargs) -> torch.Tensor:
    """Generate latent grid from prompt then decode to image."""
    img_start = tokenizer.get_special_token_ids().get("[IMG_START]", 0)
    visual_start = tokenizer.visual_start_id
    device = next(model.parameters()).device
    ids = tokenizer.encode_text(prompt) + [img_start]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(grid_size * grid_size):
        with torch.no_grad():
            logits = model.get_logits(model(input_ids))[:, -1]
        next_id = logits[:, visual_start:].argmax(dim=-1) + visual_start
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)
    grid_ids = input_ids[0, len(ids):].view(grid_size, grid_size)
    return vq_decoder.decode_from_ids(grid_ids.unsqueeze(0)) if vq_decoder else None


def generate_caption(model, tokenizer, vq_encoder, image: torch.Tensor, max_length: int = 128, **kwargs) -> str:
    """Image → Sinhala caption. Needs VQ encoder; model trained on image-caption pairs."""
    if vq_encoder is None:
        return "(Model needs VQ + image-caption training for annotation)"
    img_tokens = (vq_encoder.encode_to_ids(image) + tokenizer.visual_start_id).flatten().tolist()
    img_start = tokenizer.get_special_token_ids().get("[IMG_START]", 0)
    enc = tokenizer.encode("", add_special_tokens=False) if hasattr(tokenizer, "encode") else []
    ids = [img_start] + img_tokens + (enc if isinstance(enc, list) else list(enc))
    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    eos_id = tokenizer.get_special_token_ids().get("[EOS]", -1)
    for _ in range(max_length):
        with torch.no_grad():
            logits = model.get_logits(model(input_ids))[:, -1]
        next_id = logits.argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        if next_id.item() == eos_id:
            break
    text_ids = input_ids[0, len(ids):].tolist()
    return tokenizer.decode(text_ids, skip_special_tokens=True) if hasattr(tokenizer, "decode") else str(text_ids)


def generate_edit(model, tokenizer, vq_encoder, vq_decoder, image: torch.Tensor, instruction: str, **kwargs) -> torch.Tensor:
    """Edit image with instruction."""
    img_tokens = vq_encoder.encode_to_ids(image) + tokenizer.visual_start_id
    edit_start = tokenizer.get_special_token_ids().get("[EDIT_START]", 0)
    ids = list(img_tokens.flatten().tolist()) + [edit_start] + tokenizer.encode_text(instruction)
    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    grid_size = 16
    for _ in range(grid_size * grid_size - len(ids)):
        with torch.no_grad():
            logits = model.get_logits(model(input_ids))[:, -1]
        next_id = logits[:, tokenizer.visual_start_id:].argmax(dim=-1) + tokenizer.visual_start_id
        input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)
    out_ids = input_ids[0, -grid_size * grid_size:].view(1, grid_size, grid_size) - tokenizer.visual_start_id
    return vq_decoder.decode_from_ids(out_ids)
