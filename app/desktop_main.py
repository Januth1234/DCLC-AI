from __future__ import annotations

"""PyQt6 desktop entrypoint for DCLC Image Gen."""

import sys
from typing import Tuple, Optional

import torch
import numpy as np
from PIL import Image
from PyQt6 import QtGui, QtWidgets

from app.desktop_ui import DCLCMainWindow
from app import inference as app_inference
from app.filter import content_filter_check, filter_output


def _ensure_model_loaded() -> bool:
    """Use the shared app.inference loader so weights are only loaded once."""
    # app_inference exposes a private _load, but it's fine for internal desktop entry.
    try:
        return app_inference._load()  # type: ignore[attr-defined]
    except Exception:
        return False


def _block_if_explicit(window: DCLCMainWindow, prompt: str) -> bool:
    """Return True and show a warning if safety is ON and content looks explicit."""
    allow_unfiltered = getattr(window, "_allow_unfiltered", False)
    if content_filter_check(allow_unfiltered=allow_unfiltered, prompt=prompt):
        QtWidgets.QMessageBox.warning(
            window,
            "Safety filter active",
            "Your prompt appears to contain explicit content.\n\n"
            "With the safety layer ON (toggle off), explicit requests are blocked.",
        )
        return True
    return False


def _tensor_to_qpixmap(img: torch.Tensor) -> Optional[QtGui.QPixmap]:
    """Convert a CHW or NCHW float tensor in [0,1] to QPixmap."""
    if img is None:
        return None
    if img.dim() == 4:
        img = img[0]
    img = img.clamp(0, 1)
    arr = (img.cpu().numpy() * 255).astype("uint8")
    if arr.shape[0] in (1, 3):  # CHW → HWC
        arr = np.transpose(arr, (1, 2, 0))
    h, w = arr.shape[:2]
    if arr.shape[2] == 1:
        fmt = QtGui.QImage.Format.Format_Grayscale8
    else:
        fmt = QtGui.QImage.Format.Format_RGB888
    qimg = QtGui.QImage(arr.data, w, h, arr.strides[0], fmt)
    return QtGui.QPixmap.fromImage(qimg.copy())


def _pil_to_qpixmap(image: Image.Image) -> QtGui.QPixmap:
    """Convert a PIL image to QPixmap."""
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    data = image.tobytes("raw", "RGB")
    w, h = image.size
    qimg = QtGui.QImage(data, w, h, QtGui.QImage.Format.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg)


def _on_generate_text(window: DCLCMainWindow) -> None:
    prompt = window.text_prompt_edit.toPlainText().strip()
    if _block_if_explicit(window, prompt):
        return
    if not _ensure_model_loaded():
        window.text_output_edit.setPlainText("Model not loaded. Train first or set model_path in app_config.json.")
        return
    allow_unfiltered = getattr(window, "_allow_unfiltered", False)
    from src.inference.generator import generate_text as _gen_text

    out = _gen_text(app_inference._model, app_inference._tokenizer, prompt or "සිංහල")  # type: ignore[attr-defined]
    safe_out = filter_output(allow_unfiltered, out)
    window.text_output_edit.setPlainText(safe_out)


def _on_generate_image(window: DCLCMainWindow) -> None:
    prompt = window.image_prompt_edit.toPlainText().strip()
    if _block_if_explicit(window, prompt):
        return
    if not _ensure_model_loaded():
        window.image_label.setText("Model not loaded. Train first or set model_path in app_config.json.")
        return
    from src.inference.generator import generate_image as _gen_image

    img = _gen_image(app_inference._model, app_inference._tokenizer, app_inference._vq_dec, prompt or "රූපය")  # type: ignore[attr-defined]
    pix = _tensor_to_qpixmap(img)
    if pix is None:
        window.image_label.setText("Image decoder (VQ) not available yet for this checkpoint.")
    else:
        window.image_label.setPixmap(pix.scaled(
            window.image_label.size(),
            QtGui.Qt.AspectRatioMode.KeepAspectRatio,  # type: ignore[attr-defined]
            QtGui.Qt.TransformationMode.SmoothTransformation,  # type: ignore[attr-defined]
        ))


def _load_image_from_dialog(parent: QtWidgets.QWidget) -> Optional[Image.Image]:
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        parent,
        "Select image",
        "",
        "Images (*.png *.jpg *.jpeg *.bmp)",
    )
    if not path:
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        QtWidgets.QMessageBox.warning(parent, "Error", "Could not open image file.")
        return None


def _on_edit_load_image(window: DCLCMainWindow) -> None:
    img = _load_image_from_dialog(window)
    if img is None:
        return
    pix = _pil_to_qpixmap(img)
    window.edit_image_label.setPixmap(pix.scaled(
        window.edit_image_label.size(),
        QtGui.Qt.AspectRatioMode.KeepAspectRatio,  # type: ignore[attr-defined]
        QtGui.Qt.TransformationMode.SmoothTransformation,  # type: ignore[attr-defined]
    ))
    window._edit_source_image = img  # type: ignore[attr-defined]


def _on_apply_edit(window: DCLCMainWindow) -> None:
    instr = window.edit_instruction_edit.toPlainText().strip()
    if _block_if_explicit(window, instr):
        return
    src_img = getattr(window, "_edit_source_image", None)
    if src_img is None:
        QtWidgets.QMessageBox.information(window, "No image", "Load an image first.")
        return
    if not _ensure_model_loaded():
        window.edit_result_label.setText("Model not loaded. Train first or set model_path in app_config.json.")
        return
    from src.inference.generator import generate_edit as _gen_edit
    import torch as _torch

    arr = np.array(src_img).astype("float32") / 255.0
    tensor = _torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    out = _gen_edit(app_inference._model, app_inference._tokenizer, app_inference._vq_enc, app_inference._vq_dec, tensor, instr or "")  # type: ignore[attr-defined]
    pix = _tensor_to_qpixmap(out)
    if pix is None:
        window.edit_result_label.setText("Image edit decoder not available yet for this checkpoint.")
    else:
        window.edit_result_label.setPixmap(pix.scaled(
            window.edit_result_label.size(),
            QtGui.Qt.AspectRatioMode.KeepAspectRatio,  # type: ignore[attr-defined]
            QtGui.Qt.TransformationMode.SmoothTransformation,  # type: ignore[attr-defined]
        ))


def _on_annotate_load_image(window: DCLCMainWindow) -> None:
    img = _load_image_from_dialog(window)
    if img is None:
        return
    pix = _pil_to_qpixmap(img)
    window.ann_image_label.setPixmap(pix.scaled(
        window.ann_image_label.size(),
        QtGui.Qt.AspectRatioMode.KeepAspectRatio,  # type: ignore[attr-defined]
        QtGui.Qt.TransformationMode.SmoothTransformation,  # type: ignore[attr-defined]
    ))
    window._ann_source_image = img  # type: ignore[attr-defined]


def _on_annotate(window: DCLCMainWindow) -> None:
    img = getattr(window, "_ann_source_image", None)
    if img is None:
        QtWidgets.QMessageBox.information(window, "No image", "Load an image first.")
        return
    if not _ensure_model_loaded():
        window.ann_output_edit.setPlainText("Model not loaded. Train with image-caption data first.")
        return
    from src.inference.generator import generate_caption as _gen_caption
    import torch as _torch

    arr = np.array(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    tensor = _torch.from_numpy(arr.astype("float32") / 255.0).permute(2, 0, 1).unsqueeze(0)
    text = _gen_caption(app_inference._model, app_inference._tokenizer, app_inference._vq_enc, tensor)  # type: ignore[attr-defined]
    window.ann_output_edit.setPlainText(str(text))


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = DCLCMainWindow()

    # Wire up handlers
    win.text_generate_button.clicked.connect(lambda: _on_generate_text(win))
    win.image_generate_button.clicked.connect(lambda: _on_generate_image(win))
    win.edit_load_button.clicked.connect(lambda: _on_edit_load_image(win))
    win.edit_generate_button.clicked.connect(lambda: _on_apply_edit(win))
    win.ann_load_button.clicked.connect(lambda: _on_annotate_load_image(win))
    win.ann_generate_button.clicked.connect(lambda: _on_annotate(win))

    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

