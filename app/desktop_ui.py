from __future__ import annotations

"""PyQt6 desktop UI for DCLC Image Gen (no browser/Gradio)."""

from pathlib import Path
from typing import Optional

from PyQt6 import QtGui, QtWidgets

from app.config import load_config, save_config


class DCLCMainWindow(QtWidgets.QMainWindow):
    """Main window for the DCLC Image Gen desktop app."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._allow_unfiltered: bool = False  # session-only; always resets to filtered on launch
        self.setWindowTitle("DCLC Image Gen")
        self._init_icon()
        self._init_menu_bar()
        self._init_central_tabs()
        self.resize(900, 700)

    # --- UI construction -------------------------------------------------

    def _init_icon(self) -> None:
        """Set window icon if an icon file exists next to the app."""
        possible_paths = [
            Path(__file__).resolve().parent.parent / "build" / "dclc_image_gen.ico",
            Path(__file__).resolve().parent.parent / "build" / "dclc_image_gen_icon.png",
            Path(__file__).resolve().parent / "dclc_image_gen.ico",
            Path(__file__).resolve().parent / "dclc_image_gen_icon.png",
        ]
        for p in possible_paths:
            if p.exists():
                self.setWindowIcon(QtGui.QIcon(str(p)))
                break

    def _init_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")
        exit_action = QtGui.QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        about_action = QtGui.QAction("&About", self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)

    def _init_central_tabs(self) -> None:
        tabs = QtWidgets.QTabWidget(self)
        self.setCentralWidget(tabs)

        cfg = load_config()

        # Text tab
        self.text_prompt_edit = QtWidgets.QPlainTextEdit(self)
        self.text_prompt_edit.setPlaceholderText("Enter Sinhala prompt...")
        self.text_generate_button = QtWidgets.QPushButton("Generate", self)
        self.text_output_edit = QtWidgets.QPlainTextEdit(self)
        self.text_output_edit.setReadOnly(True)

        text_tab = QtWidgets.QWidget(self)
        text_layout = QtWidgets.QVBoxLayout(text_tab)
        text_layout.addWidget(QtWidgets.QLabel("Text prompt:", self))
        text_layout.addWidget(self.text_prompt_edit)
        text_layout.addWidget(self.text_generate_button)
        text_layout.addWidget(QtWidgets.QLabel("Output:", self))
        text_layout.addWidget(self.text_output_edit)
        tabs.addTab(text_tab, "Text")

        # Image tab
        self.image_prompt_edit = QtWidgets.QPlainTextEdit(self)
        self.image_prompt_edit.setPlaceholderText("Enter Sinhala prompt for image...")
        self.image_generate_button = QtWidgets.QPushButton("Generate", self)
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setAlignment(
            QtGui.Qt.AlignmentFlag.AlignCenter  # type: ignore[attr-defined]
        )
        self.image_label.setMinimumHeight(320)
        self.image_label.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

        image_tab = QtWidgets.QWidget(self)
        image_layout = QtWidgets.QVBoxLayout(image_tab)
        image_layout.addWidget(QtWidgets.QLabel("Image prompt:", self))
        image_layout.addWidget(self.image_prompt_edit)
        image_layout.addWidget(self.image_generate_button)
        image_layout.addWidget(QtWidgets.QLabel("Result:", self))
        image_layout.addWidget(self.image_label)
        tabs.addTab(image_tab, "Image")

        # Edit tab
        self.edit_image_label = QtWidgets.QLabel(self)
        self.edit_image_label.setAlignment(
            QtGui.Qt.AlignmentFlag.AlignCenter  # type: ignore[attr-defined]
        )
        self.edit_image_label.setMinimumHeight(240)
        self.edit_image_label.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

        self.edit_load_button = QtWidgets.QPushButton("Load image…", self)
        self.edit_instruction_edit = QtWidgets.QPlainTextEdit(self)
        self.edit_instruction_edit.setPlaceholderText("Enter edit instruction in Sinhala…")
        self.edit_generate_button = QtWidgets.QPushButton("Apply edit", self)
        self.edit_result_label = QtWidgets.QLabel(self)
        self.edit_result_label.setAlignment(
            QtGui.Qt.AlignmentFlag.AlignCenter  # type: ignore[attr-defined]
        )
        self.edit_result_label.setMinimumHeight(240)
        self.edit_result_label.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

        edit_tab = QtWidgets.QWidget(self)
        edit_layout = QtWidgets.QVBoxLayout(edit_tab)
        edit_layout.addWidget(QtWidgets.QLabel("Source image:", self))
        edit_layout.addWidget(self.edit_image_label)
        edit_layout.addWidget(self.edit_load_button)
        edit_layout.addWidget(QtWidgets.QLabel("Edit instruction:", self))
        edit_layout.addWidget(self.edit_instruction_edit)
        edit_layout.addWidget(self.edit_generate_button)
        edit_layout.addWidget(QtWidgets.QLabel("Edited image:", self))
        edit_layout.addWidget(self.edit_result_label)
        tabs.addTab(edit_tab, "Edit")

        # Annotate tab
        self.ann_image_label = QtWidgets.QLabel(self)
        self.ann_image_label.setAlignment(
            QtGui.Qt.AlignmentFlag.AlignCenter  # type: ignore[attr-defined]
        )
        self.ann_image_label.setMinimumHeight(320)
        self.ann_image_label.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

        self.ann_load_button = QtWidgets.QPushButton("Load image…", self)
        self.ann_generate_button = QtWidgets.QPushButton("Annotate", self)
        self.ann_output_edit = QtWidgets.QPlainTextEdit(self)
        self.ann_output_edit.setReadOnly(True)

        ann_tab = QtWidgets.QWidget(self)
        ann_layout = QtWidgets.QVBoxLayout(ann_tab)
        ann_layout.addWidget(QtWidgets.QLabel("Image:", self))
        ann_layout.addWidget(self.ann_image_label)
        ann_layout.addWidget(self.ann_load_button)
        ann_layout.addWidget(self.ann_generate_button)
        ann_layout.addWidget(QtWidgets.QLabel("Caption:", self))
        ann_layout.addWidget(self.ann_output_edit)
        tabs.addTab(ann_tab, "Annotate")

        # Settings tab
        self.resolution_combo = QtWidgets.QComboBox(self)
        self.resolution_combo.addItems(["128", "256"])
        # Initialise from config but never persist unfiltered between launches
        initial_res = str(cfg.get("resolution", 256))
        if initial_res in ("128", "256"):
            self.resolution_combo.setCurrentText(initial_res)
        self.safety_toggle = QtWidgets.QCheckBox("Allow unfiltered (18+) content", self)
        self.safety_toggle.setChecked(False)
        self.safety_toggle.toggled.connect(self._on_safety_toggled)
        self.settings_save_button = QtWidgets.QPushButton("Save settings", self)
        self.settings_save_button.clicked.connect(self._on_save_settings_clicked)

        settings_tab = QtWidgets.QWidget(self)
        settings_layout = QtWidgets.QFormLayout(settings_tab)
        settings_layout.addRow("Resolution:", self.resolution_combo)
        settings_layout.addRow(self.safety_toggle)
        settings_layout.addRow(self.settings_save_button)
        tabs.addTab(settings_tab, "Settings")

    # --- Dialogs ---------------------------------------------------------

    def _show_about_dialog(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "About DCLC Image Gen",
            "DCLC Image Gen\n\nOffline Sinhala text + image desktop app\npowered by the DCLC 500M model.",
        )

    # --- Settings / safety -----------------------------------------------

    def _on_safety_toggled(self, checked: bool) -> None:
        """Toggle unfiltered mode with 18+ confirmation. Off = safe (filter ON)."""
        if checked:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Confirm 18+ access",
                (
                    "Unfiltered mode may show explicit (18+) content.\n\n"
                    "By continuing you confirm you are 18+ and accept this.\n\n"
                    "Enable unfiltered mode for this session?"
                ),
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                # Keep safety layer active (toggle OFF)
                self.safety_toggle.blockSignals(True)
                self.safety_toggle.setChecked(False)
                self.safety_toggle.blockSignals(False)
                self._allow_unfiltered = False
                return
            self._allow_unfiltered = True
        else:
            # Toggle OFF → safety layer active
            self._allow_unfiltered = False

    def _on_save_settings_clicked(self) -> None:
        """Persist non-sensitive settings (e.g. resolution). Unfiltered never persisted."""
        cfg = load_config()
        try:
            cfg["resolution"] = int(self.resolution_combo.currentText())
        except ValueError:
            pass
        # Never persist allow_unfiltered; always safe-by-default on next launch
        cfg["allow_unfiltered"] = False
        save_config(cfg)

