import os
import logging
from PySide6.QtWidgets import (
    QWidget, QLabel, QComboBox, QPushButton, QGridLayout
)
from typing import Optional

logger = logging.getLogger(__name__)

class JohnsonModelSelectionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.model_label = QLabel("Johnson Model:")
        layout.addWidget(self.model_label, 0, 0)

        self.model_combo = QComboBox()
        layout.addWidget(self.model_combo, 0, 1)

        self.load_custom_model_button = QPushButton("Load Custom Johnson Model")
        layout.addWidget(self.load_custom_model_button, 0, 2, 1, 2) # Span 2 columns

    def populate_models_combo(self, models_dir: str, current_custom_model_path: Optional[str]):
        self.model_combo.clear()
        try:
            if os.path.isdir(models_dir):
                found_models = [
                    os.path.splitext(f)[0] for f in os.listdir(models_dir)
                    if f.endswith(".pth") and os.path.splitext(f)[0].lower() not in ["vgg_normalised", "decoder"]
                ]
                if found_models:
                    self.model_combo.addItems(sorted(found_models))
                else:
                    logger.info(f"No pre-defined Johnson models (.pth files) found in {models_dir}.")
            else:
                logger.warning(f"Models directory not found: {models_dir}")
        except Exception as e:
            logger.error(f"Error populating Johnson models list: {e}", exc_info=True)
            self.model_combo.addItem("Error: Could not load models")

        self.model_combo.addItem("[Use Loaded Custom Model]")

        if current_custom_model_path:
            self.model_combo.setCurrentText("[Use Loaded Custom Model]")
        elif self.model_combo.count() > 1 and self.model_combo.itemText(0) != "[Use Loaded Custom Model]":
            self.model_combo.setCurrentIndex(0)
        else:
            # Default to custom if no pre-defined or if only error/custom item exists
            self.model_combo.setCurrentText("[Use Loaded Custom Model]")

    def get_selected_model_text(self) -> str:
        return self.model_combo.currentText()

    def set_current_model_text(self, text: str):
        self.model_combo.setCurrentText(text)

    def set_controls_enabled(self, enabled: bool):
        self.model_label.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        self.load_custom_model_button.setEnabled(enabled)