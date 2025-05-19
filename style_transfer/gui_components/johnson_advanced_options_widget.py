from PySide6.QtWidgets import (
    QWidget, QLabel, QDoubleSpinBox, QCheckBox, QGridLayout, QSpinBox
)

class JohnsonAdvancedOptionsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Output Blend Alpha
        self.output_alpha_label = QLabel("Output Blend Alpha:")
        layout.addWidget(self.output_alpha_label, 0, 0)
        self.output_alpha_spinbox = QDoubleSpinBox()
        self.output_alpha_spinbox.setRange(0.0, 1.0)
        self.output_alpha_spinbox.setSingleStep(0.05)
        self.output_alpha_spinbox.setValue(1.0) # Default for Johnson output blend
        layout.addWidget(self.output_alpha_spinbox, 0, 1)

        # Generate GIF Checkbox
        self.gen_gif_checkbox = QCheckBox("Generate Johnson GIF")
        layout.addWidget(self.gen_gif_checkbox, 1, 0, 1, 2) # Span 2 cols

        # GIF Frames
        self.gif_frames_label = QLabel("GIF Frames:")
        layout.addWidget(self.gif_frames_label, 2, 0)
        self.gif_frames_spinbox = QSpinBox()
        self.gif_frames_spinbox.setRange(1, 1000)
        self.gif_frames_spinbox.setValue(20) # Default
        layout.addWidget(self.gif_frames_spinbox, 2, 1)

        # GIF Duration
        self.gif_duration_label = QLabel("GIF Duration (s):")
        layout.addWidget(self.gif_duration_label, 3, 0)
        self.gif_duration_spinbox = QDoubleSpinBox()
        self.gif_duration_spinbox.setRange(0.01, 10.0)
        self.gif_duration_spinbox.setSingleStep(0.05)
        self.gif_duration_spinbox.setDecimals(2)
        self.gif_duration_spinbox.setValue(0.1) # Default
        layout.addWidget(self.gif_duration_spinbox, 3, 1)

        # GIF Style Intensity
        self.gif_intensity_label = QLabel("GIF Style Intensity:")
        layout.addWidget(self.gif_intensity_label, 4, 0)
        self.gif_intensity_spinbox = QDoubleSpinBox()
        self.gif_intensity_spinbox.setRange(0.0, 1.0)
        self.gif_intensity_spinbox.setSingleStep(0.05)
        self.gif_intensity_spinbox.setValue(1.0) # Default
        layout.addWidget(self.gif_intensity_spinbox, 4, 1)

        # Ping-Pong GIF Checkbox
        self.ping_pong_checkbox = QCheckBox("Ping-Pong Effect")
        layout.addWidget(self.ping_pong_checkbox, 5, 0, 1, 2) # Span 2 cols

        # Connect signals
        self.gen_gif_checkbox.toggled.connect(self.update_gif_controls_enabled_state)

        # Initial state update for GIF sub-controls
        self.update_gif_controls_enabled_state(self.gen_gif_checkbox.isChecked())

    def get_output_blend_alpha(self) -> float:
        return self.output_alpha_spinbox.value()

    def get_generate_gif(self) -> bool:
        return self.gen_gif_checkbox.isChecked()

    def get_gif_frames(self) -> int:
        return self.gif_frames_spinbox.value()

    def get_gif_duration(self) -> float:
        return self.gif_duration_spinbox.value()

    def get_gif_style_intensity(self) -> float:
        return self.gif_intensity_spinbox.value()

    def get_gif_ping_pong(self) -> bool:
        return self.ping_pong_checkbox.isChecked()

    def update_gif_controls_enabled_state(self, checked: bool):
        """Enable/disable GIF-specific controls based on the 'Generate GIF' checkbox."""
        enable = checked
        self.gif_frames_label.setEnabled(enable)
        self.gif_frames_spinbox.setEnabled(enable)
        self.gif_duration_label.setEnabled(enable)
        self.gif_duration_spinbox.setEnabled(enable)
        self.gif_intensity_label.setEnabled(enable)
        self.gif_intensity_spinbox.setEnabled(enable)
        self.ping_pong_checkbox.setEnabled(enable)

    def set_controls_enabled(self, enabled: bool):
        """Enable or disable all controls in this widget group."""
        self.output_alpha_label.setEnabled(enabled)
        self.output_alpha_spinbox.setEnabled(enabled)
        self.gen_gif_checkbox.setEnabled(enabled)

        if enabled:
            self.update_gif_controls_enabled_state(self.gen_gif_checkbox.isChecked())
        else:
            self.gif_frames_label.setEnabled(False)
            self.gif_frames_spinbox.setEnabled(False)
            self.gif_duration_label.setEnabled(False)
            self.gif_duration_spinbox.setEnabled(False)
            self.gif_intensity_label.setEnabled(False)
            self.gif_intensity_spinbox.setEnabled(False)
            self.ping_pong_checkbox.setEnabled(False)