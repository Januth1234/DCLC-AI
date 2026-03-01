"""Translate English captions to Sinhala (bootstrap)."""
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def main(input_json: str, output_json: str, batch_size: int = 8):
    """Load captions, translate with NLLB/mT5, save."""
    path = Path(input_json)
    if not path.exists():
        logging.warning("Input %s not found", input_json)
        return
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        samples = data
    else:
        samples = list(data.values()) if isinstance(data, dict) else []
    try:
        from transformers import pipeline
        pipe = pipeline("translation", model="facebook/nllb-200-1.3B", src_lang="eng_Latn", tgt_lang="sin_Sinh")
    except Exception as e:
        logging.warning("Translation model not available: %s. Save without translation.", e)
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(samples, f, ensure_ascii=False)
        return
    out = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        caps = [s.get("caption", s.get("text", "")) for s in batch]
        trans = pipe(caps)
        for s, t in zip(batch, trans):
            s["sinhala_caption"] = t[0]["translation_text"] if isinstance(t, list) else t.get("translation_text", str(t))
            out.append(s)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/image_captions.json")
    p.add_argument("--output", default="data/sinhala_image_captions.json")
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()
    main(a.input, a.output, a.batch_size)
