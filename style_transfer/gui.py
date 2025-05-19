import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
    QComboBox, QDoubleSpinBox, QCheckBox, QLineEdit, QMessageBox, QProgressBar, QGridLayout, QGroupBox, QSpinBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QMovie
import os # For creating default output path
import tempfile # For default output
import logging # Added for logger
from pathlib import Path # Added for dummy file creation

# Assuming cli.py and its functions are accessible
# This might need adjustment based on your project structure if gui.py is moved
from style_transfer.cli import run_style_transfer_pipeline
from style_transfer.utils.device_utils import get_available_devices, clear_gpu_memory
from style_transfer.gui_components.style_transfer_thread import StyleTransferThread, DEFAULT_ALPHA, DEFAULT_ADAIN_GIF_FRAMES, DEFAULT_ADAIN_GIF_DURATION
from style_transfer.gui_components.adain_options_widget import AdaINOptionsWidget # Import new widget
from style_transfer.gui_components.johnson_model_selection_widget import JohnsonModelSelectionWidget # Import new widget
from style_transfer.gui_components.johnson_advanced_options_widget import JohnsonAdvancedOptionsWidget # Import new widget

logger = logging.getLogger(__name__) # Added logger instance

IMAGE_DISPLAY_SIZE = 256
# DEFAULT_ALPHA = 1.0 # Moved to style_transfer_thread.py
DEFAULT_MAX_SIZE = 512 # This is used by MainWindow.run_transfer, so keep or pass to thread
# DEFAULT_ADAIN_GIF_FRAMES = 20 # Moved
# DEFAULT_ADAIN_GIF_DURATION = 0.1 # Moved

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Style Transfer GUI")
        self.setGeometry(100, 100, IMAGE_DISPLAY_SIZE * 4 + 200, IMAGE_DISPLAY_SIZE + 400) # Adjusted size

        self.content_image_path = None
        self.style_image_path = None
        self.output_image_path = None # To store the path of the generated image
        self.generated_gif_path = None # To store path of generated GIF
        self.custom_johnson_model_path = None # Added to store custom model path
        self.default_output_dir = os.path.join(tempfile.gettempdir(), "style_transfer_outputs")
        os.makedirs(self.default_output_dir, exist_ok=True)
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.project_root, "models")

        self.thread = None # For running style transfer in background
        self.gif_movie = None # For QMovie instance

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Image Display Area ---
        image_display_layout = QHBoxLayout()
        self.content_display_label = self.create_image_display_label("Content Image")
        self.style_display_label = self.create_image_display_label("Style Image (AdaIN)") # Clarify usage
        self.output_display_label = self.create_image_display_label("Output Image")
        self.gif_display_label = self.create_image_display_label("Animated GIF") # New GIF display
        image_display_layout.addWidget(self.content_display_label)
        image_display_layout.addWidget(self.style_display_label)
        image_display_layout.addWidget(self.output_display_label)
        image_display_layout.addWidget(self.gif_display_label) # Add to layout
        main_layout.addLayout(image_display_layout)

        # --- File Selection Layout ---
        file_selection_layout = QHBoxLayout()

        # Content Image Selection
        content_selection_vbox = QVBoxLayout()
        self.content_image_label = QLabel("No content image selected.")
        self.content_image_label.setAlignment(Qt.AlignCenter)
        content_selection_vbox.addWidget(self.content_image_label)
        content_button = QPushButton("Select Content Image")
        content_button.clicked.connect(self.select_content_image)
        content_selection_vbox.addWidget(content_button)
        file_selection_layout.addLayout(content_selection_vbox)

        # Style Image Selection
        style_selection_vbox = QVBoxLayout()
        self.style_image_label = QLabel("No style image selected (for AdaIN).")
        self.style_image_label.setAlignment(Qt.AlignCenter)
        style_selection_vbox.addWidget(self.style_image_label)
        style_button = QPushButton("Select Style Image (AdaIN)")
        style_button.setObjectName("selectStyleImageButtonAdaIN")
        style_button.clicked.connect(self.select_style_image)
        style_selection_vbox.addWidget(style_button)
        file_selection_layout.addLayout(style_selection_vbox)
        main_layout.addLayout(file_selection_layout)

        # --- Controls Area ---
        controls_layout = QGridLayout() # Using QGridLayout for better organization

        # Method Selection
        controls_layout.addWidget(QLabel("Method:"), 0, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["AdaIN", "Johnson"])
        self.method_combo.currentTextChanged.connect(self.update_controls_for_method)
        controls_layout.addWidget(self.method_combo, 0, 1)
        # self.method_combo.setCurrentText("AdaIN") # Deferred: will set and call update explicitly later

        # Device Selection
        controls_layout.addWidget(QLabel("Device:"), 0, 2)
        self.device_combo = QComboBox()
        self.populate_device_combo()
        controls_layout.addWidget(self.device_combo, 0, 3)

        # --- AdaIN Controls (Refactored) ---
        self.adain_options_widget = AdaINOptionsWidget()
        controls_layout.addWidget(self.adain_options_widget, 1, 0, 1, 4) # Span 4 columns

        # --- Johnson Model Selection Controls (Refactored) ---
        self.johnson_model_selection_widget = JohnsonModelSelectionWidget()
        self.johnson_model_selection_widget.load_custom_model_button.clicked.connect(self.select_johnson_model_weights)
        self.johnson_model_selection_widget.populate_models_combo(self.models_dir, self.custom_johnson_model_path)
        controls_layout.addWidget(self.johnson_model_selection_widget, 2, 0, 1, 4)

        # --- Johnson Advanced Options (Refactored) ---
        self.johnson_advanced_options_widget = JohnsonAdvancedOptionsWidget()
        controls_layout.addWidget(self.johnson_advanced_options_widget, 3, 0, 1, 4)

        main_layout.addLayout(controls_layout)

        # Set default method and update UI accordingly AFTER all controls are in layouts
        self.method_combo.setCurrentText("AdaIN")
        self.update_controls_for_method("AdaIN") # Explicitly update to show only AdaIN controls

        # These calls ensure sub-controls (like GIF frames/duration) start disabled
        # regardless of the main GIF checkbox state, which is handled by update_controls_for_method.
        self.update_johnson_gif_controls_enabled_state(self.johnson_advanced_options_widget.get_generate_gif())

        # --- Run Button ---
        self.run_button = QPushButton("Run Style Transfer")
        self.run_button.clicked.connect(self.run_transfer)
        main_layout.addWidget(self.run_button)

        # --- Progress Bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False) # Hide initially
        main_layout.addWidget(self.progress_bar)

        # --- Save Output Button ---
        self.save_output_button = QPushButton("Save Output Image")
        self.save_output_button.clicked.connect(self.save_output_image)
        self.save_output_button.setEnabled(False) # Disabled until output is generated
        main_layout.addWidget(self.save_output_button)

        self.save_gif_button = QPushButton("Save Animated GIF")
        self.save_gif_button.clicked.connect(self.save_generated_gif)
        self.save_gif_button.setEnabled(False) # Disabled until GIF is generated
        main_layout.addWidget(self.save_gif_button)

        main_layout.addStretch()

    def populate_device_combo(self):
        devices = get_available_devices()
        self.device_combo.addItems(devices)
        if "cuda" in devices:
            self.device_combo.setCurrentText("cuda")
        elif "mps" in devices: # For MacOS Metal
             self.device_combo.setCurrentText("mps")
        else:
            self.device_combo.setCurrentText("cpu")

    def update_controls_for_method(self, method_name):
        is_adain = (method_name == "AdaIN")
        is_johnson = (method_name == "Johnson")

        self.adain_options_widget.setVisible(is_adain)
        self.adain_options_widget.set_controls_enabled(is_adain) # Enable/disable all controls in the widget
        self.style_display_label.setVisible(is_adain)

        style_button = self.findChild(QPushButton, "selectStyleImageButtonAdaIN")
        if style_button:
            style_button.setVisible(is_adain)

        self.johnson_model_selection_widget.setVisible(is_johnson)
        self.johnson_model_selection_widget.set_controls_enabled(is_johnson)
        self.johnson_advanced_options_widget.setVisible(is_johnson)
        self.johnson_advanced_options_widget.set_controls_enabled(is_johnson)

        if is_johnson:
            # This call is implicitly handled by johnson_advanced_options_widget.set_controls_enabled(True)
            # which calls its internal update_gif_controls_enabled_state based on its own checkbox state.
            pass
        else:
            # If not Johnson, ensure its GIF controls (within the widget) are fully disabled.
            # This is handled by johnson_advanced_options_widget.set_controls_enabled(False)
            pass

        if is_johnson:
            self.style_display_label.setText("Style Image (Not used for Johnson)")
            self.style_display_label.setPixmap(QPixmap())
        else:
            self.style_display_label.setText("Style Image (AdaIN)")
            if self.style_image_path:
                 pixmap = QPixmap(self.style_image_path)
                 self.style_display_label.setPixmap(pixmap.scaled(
                    IMAGE_DISPLAY_SIZE, IMAGE_DISPLAY_SIZE,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
            else:
                self.style_display_label.setText("Style Image (AdaIN)")

    def update_johnson_gif_controls_enabled_state(self, checked):
        is_johnson_method_actually_selected = (self.method_combo.currentText() == "Johnson")
        enable_gif_widgets = checked and is_johnson_method_actually_selected

        self.johnson_advanced_options_widget.set_controls_enabled(enable_gif_widgets)

    def create_image_display_label(self, placeholder_text):
        label = QLabel(placeholder_text)
        label.setFixedSize(IMAGE_DISPLAY_SIZE, IMAGE_DISPLAY_SIZE)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("border: 1px solid gray;") # Add a border for visibility
        return label

    def select_content_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Content Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.content_image_path = file_path
            self.content_image_label.setText(f"Content: {Path(file_path).name}")
            pixmap = QPixmap(file_path)
            self.content_display_label.setPixmap(pixmap.scaled(
                IMAGE_DISPLAY_SIZE, IMAGE_DISPLAY_SIZE,
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

    def select_style_image(self):
        if self.method_combo.currentText() != "AdaIN":
            QMessageBox.information(self, "Info", "Style image is only used for AdaIN method.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Style Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.style_image_path = file_path
            self.style_image_label.setText(f"Style: {Path(file_path).name}")
            pixmap = QPixmap(file_path)
            self.style_display_label.setPixmap(pixmap.scaled(
                IMAGE_DISPLAY_SIZE, IMAGE_DISPLAY_SIZE,
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

    def select_johnson_model_weights(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Johnson Model Weights",
            "", # Start in current/last directory
            "PyTorch Models (*.pth *.pt)"
        )
        if file_path:
            self.custom_johnson_model_path = file_path
            self.johnson_model_selection_widget.set_current_model_text("[Use Loaded Custom Model]")
            QMessageBox.information(self, "Model Loaded", f"Custom Johnson model loaded: {Path(file_path).name}. Select '[Use Loaded Custom Model]' from the dropdown to use it.")

    def run_transfer(self):
        if not self.content_image_path:
            QMessageBox.warning(self, "Missing Input", "Please select a content image.")
            return

        current_method = self.method_combo.currentText()

        # Prepare parameters for the thread
        thread_params = {
            "device": self.device_combo.currentText(),
            "max_size": DEFAULT_MAX_SIZE # Or make this configurable later
        }

        # Define a unique output filename
        base_name_content = os.path.splitext(os.path.basename(self.content_image_path))[0]
        output_filename_parts = [base_name_content, "styled"]

        current_style_image_path = self.style_image_path # This can be None
        gif_requested_this_run = False # Initialize flag

        if current_method == "AdaIN":
            if not current_style_image_path:
                QMessageBox.warning(self, "Missing Input", "Please select a style image for AdaIN.")
                return
            thread_params["alpha"] = self.adain_options_widget.get_alpha()
            thread_params["preserve_color"] = self.adain_options_widget.get_preserve_color()
            thread_params["generate_adain_alpha_sequence"] = self.adain_options_widget.get_generate_gif()

            base_name_style = os.path.splitext(os.path.basename(current_style_image_path))[0]
            output_filename_parts.extend(["with", base_name_style, "adain"])

            if thread_params["generate_adain_alpha_sequence"]:
                gif_requested_this_run = True
                thread_params["adain_sequence_num_frames"] = self.adain_options_widget.get_gif_frames()
                thread_params["adain_gif_frame_duration"] = self.adain_options_widget.get_gif_duration()
                thread_params["adain_gif_ping_pong"] = self.adain_options_widget.get_gif_ping_pong()
                adain_gif_filename = "_".join(output_filename_parts) + "_sequence.gif"
                thread_params["adain_sequence_gif_path"] = os.path.join(self.default_output_dir, adain_gif_filename)

        elif current_method == "Johnson":
            selected_johnson_model_text = self.johnson_model_selection_widget.get_selected_model_text()
            johnson_model_weights_path = None
            model_short_name = ""

            if selected_johnson_model_text == "[Use Loaded Custom Model]":
                if not self.custom_johnson_model_path:
                    QMessageBox.warning(self, "Missing Input", "No custom Johnson model has been loaded. Please use the 'Load Custom Johnson Model' button first.")
                    return
                johnson_model_weights_path = self.custom_johnson_model_path
                model_short_name = "custom_" + os.path.splitext(os.path.basename(self.custom_johnson_model_path))[0]
            else: # It's a model name from the models/ directory
                if not selected_johnson_model_text or selected_johnson_model_text == "Error: Could not load models":
                     QMessageBox.warning(self, "Missing Input", f"Please select a valid Johnson model from the dropdown or load a custom one.")
                     return
                model_name_from_combo = selected_johnson_model_text
                potential_model_path = os.path.join(self.models_dir, model_name_from_combo + ".pth")
                if not os.path.exists(potential_model_path):
                    QMessageBox.critical(self, "Error", f"Johnson model file not found: {potential_model_path}. Try refreshing or loading a custom model.")
                    return
                johnson_model_weights_path = potential_model_path
                model_short_name = model_name_from_combo

            if not johnson_model_weights_path:
                QMessageBox.warning(self, "Missing Input", f"Please select or load a Johnson model.")
                return

            thread_params["johnson_model_weights"] = johnson_model_weights_path
            thread_params["johnson_output_blend_alpha"] = self.johnson_advanced_options_widget.get_output_blend_alpha()
            thread_params["generate_johnson_gif"] = self.johnson_advanced_options_widget.get_generate_gif()

            if thread_params["generate_johnson_gif"]:
                gif_requested_this_run = True
                thread_params["johnson_gif_frames"] = self.johnson_advanced_options_widget.get_gif_frames()
                thread_params["johnson_gif_duration"] = self.johnson_advanced_options_widget.get_gif_duration()
                thread_params["johnson_gif_style_intensity"] = self.johnson_advanced_options_widget.get_gif_style_intensity()
                thread_params["johnson_gif_ping_pong"] = self.johnson_advanced_options_widget.get_gif_ping_pong()
                johnson_gif_filename = "_".join(output_filename_parts) + "_animated.gif"
                thread_params["johnson_gif_path"] = os.path.join(self.default_output_dir, johnson_gif_filename)
            else:
                thread_params["johnson_gif_frames"] = 20
                thread_params["johnson_gif_duration"] = 0.1
                thread_params["johnson_gif_style_intensity"] = 1.0

            output_filename_parts.extend(["johnson", model_short_name])
        else:
            QMessageBox.critical(self, "Error", "Invalid method selected.")
            return

        output_filename = "_".join(output_filename_parts) + ".jpg"
        self.output_image_path = os.path.join(self.default_output_dir, output_filename)

        self.run_button.setEnabled(False)
        self.save_output_button.setEnabled(False)
        self.output_display_label.setText("Processing...")
        self.gif_display_label.setText("GIF Processing..." if gif_requested_this_run else "Animated GIF")
        if self.gif_movie and self.gif_movie.isValid(): self.gif_movie.stop()
        self.gif_display_label.setMovie(None)
        self.save_gif_button.setEnabled(False)

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0,0) # Indeterminate progress

        self.thread = StyleTransferThread(
            self.content_image_path,
            current_style_image_path,
            self.output_image_path,
            current_method,
            thread_params
        )
        self.thread.finished_signal.connect(self.on_transfer_finished)
        self.thread.start()

    def on_transfer_finished(self, main_output_path, gif_output_path, error_message):
        self.run_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0,1) # Reset progress bar

        if error_message:
            logger.error(f"GUI received error from thread: {error_message}") # Log it too
            QMessageBox.critical(self, "Error", f"Style transfer failed: {error_message}")
            self.output_display_label.setText("Error!")
            self.gif_display_label.setText("GIF Error!" if self.generated_gif_path else "Animated GIF")
            self.output_image_path = None # Clear output path on error
            self.generated_gif_path = None # Clear on error too
            self.save_output_button.setEnabled(False) # Disable save button on error
            self.save_gif_button.setEnabled(False) # Disable save GIF button on error
        elif main_output_path:
            self.output_image_path = main_output_path # Store the actual output path
            pixmap = QPixmap(main_output_path)
            self.output_display_label.setPixmap(pixmap.scaled(
                IMAGE_DISPLAY_SIZE, IMAGE_DISPLAY_SIZE,
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            self.save_output_button.setEnabled(True)
            QMessageBox.information(self, "Success", f"Style transfer complete! Output saved to: {main_output_path}")

            if gif_output_path and Path(gif_output_path).exists():
                self.generated_gif_path = gif_output_path # Store it
                self.gif_movie = QMovie(gif_output_path)
                if self.gif_movie.isValid():
                    self.gif_display_label.setMovie(self.gif_movie)
                    self.gif_movie.setScaledSize(self.gif_display_label.size()*.9) # Slightly smaller
                    self.gif_movie.start()
                    logger.info(f"Successfully loaded and started GIF: {gif_output_path}")
                    self.save_gif_button.setEnabled(True) # Enable save GIF button
                else:
                    logger.error(f"QMovie could not load GIF: {gif_output_path}. Is it a valid GIF?")
                    self.gif_display_label.setText("Invalid GIF File")
                    self.save_gif_button.setEnabled(False)
            elif self.adain_options_widget.get_generate_gif() or self.johnson_advanced_options_widget.get_generate_gif(): # If GIF was expected but not produced
                logger.warning(f"GIF was requested but not found at path: {gif_output_path}")
                self.gif_display_label.setText("GIF Not Found")
                self.save_gif_button.setEnabled(False)
            else: # No GIF requested or generated
                self.gif_display_label.setText("Animated GIF")
                if self.gif_movie and self.gif_movie.isValid(): self.gif_movie.stop()
                self.gif_display_label.setMovie(None)
                self.save_gif_button.setEnabled(False)
        else:
            self.output_display_label.setText("Output Image") # Reset placeholder
            self.output_image_path = None
            self.generated_gif_path = None
            self.save_output_button.setEnabled(False)
            self.save_gif_button.setEnabled(False)

    def save_output_image(self):
        if not self.output_image_path or not os.path.exists(self.output_image_path):
            QMessageBox.warning(self, "No Output", "No output image to save or output path is invalid.")
            return

        # Suggest a filename based on the generated one
        suggested_filename = os.path.basename(self.output_image_path)

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Stylized Image",
            suggested_filename, # Default filename in dialog
            "Images (*.png *.jpg *.jpeg)"
        )
        if save_path:
            try:
                # QPixmap can save itself, or we can copy the file
                # For simplicity, let's copy the file as run_style_transfer_pipeline already saved it.
                import shutil
                shutil.copy(self.output_image_path, save_path)
                QMessageBox.information(self, "Saved", f"Image saved to {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save image: {e}")

    def save_generated_gif(self):
        if not self.generated_gif_path or not os.path.exists(self.generated_gif_path):
            QMessageBox.warning(self, "No GIF", "No GIF available to save or the GIF path is invalid.")
            return

        suggested_filename = os.path.basename(self.generated_gif_path)

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Animated GIF",
            suggested_filename, # Default filename in dialog
            "Animated GIF (*.gif)"
        )
        if save_path:
            try:
                import shutil
                shutil.copy(self.generated_gif_path, save_path)
                QMessageBox.information(self, "GIF Saved", f"Animated GIF saved to {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save GIF: {e}")

    def closeEvent(self, event):
        # Clean up the thread if it's running
        if self.thread and self.thread.isRunning():
            # self.thread.quit() # Request interruption
            self.thread.terminate() # Forcefully terminate if quit doesn't work quickly
            self.thread.wait()  # Wait for thread to finish

        # Clean up temporary output directory if you want, or leave it for user
        # For example, to remove it:
        # if os.path.exists(self.default_output_dir):
        #     import shutil
        #     try:
        #         shutil.rmtree(self.default_output_dir)
        #     except Exception as e:
        #         print(f"Could not remove temp dir {self.default_output_dir}: {e}")
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()