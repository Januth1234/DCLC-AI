"""Gradio UI for DCLC."""
import gradio as gr
from app.config import load_config, save_config
from app.inference import generate_text, generate_image, generate_edit, generate_caption


def build_ui():
    cfg = load_config()
    with gr.Blocks(title="DCLC") as demo:
        try:
            from src.version import get_version
            ver = get_version()
        except Exception:
            ver = "0.1.0"
        gr.Markdown("# DCLC - Dual-Core Latent Composer (v%s)" % ver)
        with gr.Tabs():
            with gr.Tab("Text"):
                text_in = gr.Textbox(label="Sinhala prompt", lines=3)
                text_btn = gr.Button("Generate")
                text_out = gr.Textbox(label="Output", lines=10)
                text_btn.click(fn=lambda p: generate_text(p or "සිංහල"), inputs=[text_in], outputs=[text_out])
            with gr.Tab("Image"):
                img_prompt = gr.Textbox(label="Sinhala prompt")
                img_btn = gr.Button("Generate")
                img_out = gr.Image(label="Output")
                img_btn.click(fn=generate_image, inputs=[img_prompt], outputs=[img_out])
            with gr.Tab("Edit"):
                edit_img = gr.Image(label="Image")
                edit_instr = gr.Textbox(label="Edit instruction (Sinhala)")
                edit_btn = gr.Button("Edit")
                edit_out = gr.Image(label="Output")
                edit_btn.click(fn=lambda img, i: generate_edit(img, i) if img else None, inputs=[edit_img, edit_instr], outputs=[edit_out])
            with gr.Tab("Annotate"):
                ann_img = gr.Image(label="Upload image → Sinhala caption")
                ann_btn = gr.Button("Annotate")
                ann_out = gr.Textbox(label="Caption (සිංහල)", lines=4)
                ann_btn.click(fn=generate_caption, inputs=[ann_img], outputs=[ann_out])
            with gr.Tab("Settings"):
                filter_toggle = gr.Checkbox(label="Content filtering (OFF = unfiltered, default)", value=not cfg.get("allow_unfiltered", True))
                res_drop = gr.Dropdown(choices=["128", "256"], value=str(cfg.get("resolution", 256)), label="Resolution")
                def save_s(checked, res):
                    save_config({"allow_unfiltered": not checked, "resolution": int(res)})
                gr.Button("Save").click(fn=save_s, inputs=[filter_toggle, res_drop])
        return demo


if __name__ == "__main__":
    build_ui().launch(share=False)
