from PySide6.QtWidgets import (
    QWidget, QLabel, QDoubleSpinBox, QCheckBox, QGridLayout, QSpinBox
)

# It's good practice to have constants in a central place if shared across many GUI components.
# For now, importing from style_transfer_thread is fine, but consider a dedicated constants.py later.
from .style_transfer_thread import DEFAULT_ALPHA, DEFAULT_ADAIN_GIF_FRAMES, DEFAULT_ADAIN_GIF_DURATION

class AdaINOptionsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Alpha
        self.alpha_label = QLabel("Alpha:")
        layout.addWidget(self.alpha_label, 0, 0)
        self.alpha_spinbox = QDoubleSpinBox()
        self.alpha_spinbox.setRange(0.0, 1.0)
        self.alpha_spinbox.setSingleStep(0.1)
        self.alpha_spinbox.setValue(DEFAULT_ALPHA)
        layout.addWidget(self.alpha_spinbox, 0, 1)

        # Preserve Color
        self.preserve_color_checkbox = QCheckBox("Preserve Color")
        layout.addWidget(self.preserve_color_checkbox, 0, 2, 1, 2) # Span 2 columns

        # Generate GIF Checkbox
        self.gen_gif_checkbox = QCheckBox("Generate AdaIN GIF")
        layout.addWidget(self.gen_gif_checkbox, 1, 0, 1, 2) # Span 2 columns

        # GIF Frames
        self.gif_frames_label = QLabel("GIF Frames:")
        layout.addWidget(self.gif_frames_label, 2, 0)
        self.gif_frames_spinbox = QSpinBox()
        self.gif_frames_spinbox.setRange(1, 1000)
        self.gif_frames_spinbox.setValue(DEFAULT_ADAIN_GIF_FRAMES)
        layout.addWidget(self.gif_frames_spinbox, 2, 1)

        # GIF Duration
        self.gif_duration_label = QLabel("GIF Duration (s):")
        layout.addWidget(self.gif_duration_label, 2, 2)
        self.gif_duration_spinbox = QDoubleSpinBox()
        self.gif_duration_spinbox.setRange(0.01, 10.0)
        self.gif_duration_spinbox.setSingleStep(0.05)
        self.gif_duration_spinbox.setDecimals(2)
        self.gif_duration_spinbox.setValue(DEFAULT_ADAIN_GIF_DURATION)
        layout.addWidget(self.gif_duration_spinbox, 2, 3)

        # Ping-Pong GIF Checkbox
        self.ping_pong_checkbox = QCheckBox("Ping-Pong Effect")
        layout.addWidget(self.ping_pong_checkbox, 3, 0, 1, 2) # Span 2 columns

        # Connect signals
        self.gen_gif_checkbox.toggled.connect(self.update_gif_controls_enabled_state)

        # Initial state update for GIF sub-controls
        self.update_gif_controls_enabled_state(self.gen_gif_checkbox.isChecked())

    def get_alpha(self) -> float:
        return self.alpha_spinbox.value()

    def get_preserve_color(self) -> bool:
        return self.preserve_color_checkbox.isChecked()

    def get_generate_gif(self) -> bool:
        return self.gen_gif_checkbox.isChecked()

    def get_gif_frames(self) -> int:
        return self.gif_frames_spinbox.value()

    def get_gif_duration(self) -> float:
        return self.gif_duration_spinbox.value()

    def get_gif_ping_pong(self) -> bool:
        return self.ping_pong_checkbox.isChecked()

    def update_gif_controls_enabled_state(self, checked: bool):
        """Enable/disable GIF-specific controls based on the 'Generate GIF' checkbox."""
        enable = checked
        self.gif_frames_label.setEnabled(enable)
        self.gif_frames_spinbox.setEnabled(enable)
        self.gif_duration_label.setEnabled(enable)
        self.gif_duration_spinbox.setEnabled(enable)
        self.ping_pong_checkbox.setEnabled(enable)

    def set_controls_enabled(self, enabled: bool):
        """Enable or disable all controls in this widget group."""
        # This method is intended to be called by MainWindow when switching methods.
        # It ensures that even if the main "Generate GIF" checkbox is checked,
        # the sub-controls are only enabled if the entire AdaIN group is active.
        self.alpha_label.setEnabled(enabled)
        self.alpha_spinbox.setEnabled(enabled)
        self.preserve_color_checkbox.setEnabled(enabled)
        self.gen_gif_checkbox.setEnabled(enabled)

        # For GIF sub-controls, their state depends on BOTH the overall group being enabled
        # AND the 'gen_gif_checkbox' being checked.
        if enabled:
            self.update_gif_controls_enabled_state(self.gen_gif_checkbox.isChecked())
        else:
            # If the whole group is disabled, all sub-controls must be disabled.
            self.gif_frames_label.setEnabled(False)
            self.gif_frames_spinbox.setEnabled(False)
            self.gif_duration_label.setEnabled(False)
            self.gif_duration_spinbox.setEnabled(False)
            self.ping_pong_checkbox.setEnabled(False)