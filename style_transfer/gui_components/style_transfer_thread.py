from PySide6.QtCore import QThread, Signal
from pathlib import Path
import tempfile
import logging

# Assuming cli.py and its functions are accessible from the original gui.py location
# This might need adjustment if the project structure changes how cli is imported.
# For now, assume it can be imported relative to the style_transfer package root.
from ..cli import run_style_transfer_pipeline # Relative import
from ..utils.device_utils import clear_gpu_memory # Relative import

logger = logging.getLogger(__name__)

DEFAULT_MAX_SIZE = 512  # Consider moving constants to a shared location if used by more components
DEFAULT_ALPHA = 1.0
DEFAULT_ADAIN_GIF_FRAMES = 20
DEFAULT_ADAIN_GIF_DURATION = 0.1


class StyleTransferThread(QThread):
    finished_signal = Signal(object, object, str) # main_output_path, gif_output_path, error_message

    def __init__(self, content_path, style_path, output_path, method, params):
        super().__init__()
        self.content_path = content_path
        self.style_path = style_path # Can be None if method is Johnson & no style selected
        self.output_path = output_path
        self.method = method
        self.params = params
        # Store GIF paths directly if they are in params, as they are determined by MainWindow
        self.adain_gif_path = params.get("adain_sequence_gif_path")
        self.johnson_gif_path = params.get("johnson_gif_path")


    def run(self):
        main_output_path_to_emit = None
        gif_output_path_to_emit = None # Initialize
        error_message_to_emit = None
        try:
            pipeline_args = {
                "content_path_str": self.content_path,
                "output_path_str": self.output_path,
                "max_size": self.params.get("max_size", DEFAULT_MAX_SIZE),
                "device_name": self.params.get("device"),
            }

            current_style_path = self.style_path

            if self.method == "AdaIN":
                if not current_style_path:
                    error_message_to_emit = "Style image is required for AdaIN method."
                    self.finished_signal.emit(None, None, error_message_to_emit)
                    return
                pipeline_args["model_type"] = "adain"
                pipeline_args["alpha"] = self.params.get("alpha", DEFAULT_ALPHA)
                pipeline_args["preserve_color"] = self.params.get("preserve_color", False)
                pipeline_args["style_path_str"] = current_style_path
                # AdaIN GIF params
                pipeline_args["generate_adain_alpha_sequence"] = self.params.get("generate_adain_alpha_sequence", False)
                if pipeline_args["generate_adain_alpha_sequence"]:
                    pipeline_args["adain_sequence_num_frames"] = self.params.get("adain_sequence_num_frames", DEFAULT_ADAIN_GIF_FRAMES)
                    pipeline_args["adain_gif_frame_duration"] = self.params.get("adain_gif_frame_duration", DEFAULT_ADAIN_GIF_DURATION)
                    pipeline_args["adain_gif_ping_pong"] = self.params.get("adain_gif_ping_pong", False)
                    # Use the adain_gif_path stored in self for pipeline_args and for emitting
                    if self.adain_gif_path:
                        pipeline_args["adain_sequence_gif_path"] = self.adain_gif_path
                        gif_output_path_to_emit = self.adain_gif_path
                    else:
                        pipeline_args["adain_sequence_gif_path"] = None

            elif self.method == "Johnson":
                pipeline_args["model_type"] = "johnson"
                pipeline_args["johnson_model_weights"] = self.params.get("johnson_model_weights")
                if not pipeline_args["johnson_model_weights"]:
                    error_message_to_emit = "Johnson model weights path must be provided."
                    self.finished_signal.emit(None, None, error_message_to_emit)
                    return

                pipeline_args["johnson_output_blend_alpha"] = self.params.get("johnson_output_blend_alpha", 1.0)
                pipeline_args["generate_johnson_gif"] = self.params.get("generate_johnson_gif", False)

                if pipeline_args["generate_johnson_gif"]:
                    pipeline_args["johnson_gif_frames"] = self.params.get("johnson_gif_frames", 20)
                    pipeline_args["johnson_gif_duration"] = self.params.get("johnson_gif_duration", 0.1)
                    pipeline_args["johnson_gif_style_intensity"] = self.params.get("johnson_gif_style_intensity", 1.0)
                    pipeline_args["johnson_gif_ping_pong"] = self.params.get("johnson_gif_ping_pong", False)
                    # Use the johnson_gif_path stored in self for pipeline_args and for emitting
                    if self.johnson_gif_path:
                        pipeline_args["johnson_gif_path"] = self.johnson_gif_path
                        gif_output_path_to_emit = self.johnson_gif_path
                    else:
                        pipeline_args["johnson_gif_path"] = None
                else:
                    pipeline_args["johnson_gif_frames"] = 20
                    pipeline_args["johnson_gif_duration"] = 0.1
                    pipeline_args["johnson_gif_style_intensity"] = 1.0
                    pipeline_args["johnson_gif_ping_pong"] = False
                    pipeline_args["johnson_gif_path"] = None

                pipeline_args["style_path_str"] = current_style_path if current_style_path else str(Path(tempfile.gettempdir()) / "dummy_style_for_johnson.png")

            else:
                error_message_to_emit = f"Unsupported method: {self.method}"
                self.finished_signal.emit(None, None, error_message_to_emit)
                return

            logger.info(f"Running pipeline with args: {pipeline_args}")
            run_style_transfer_pipeline(**pipeline_args)
            main_output_path_to_emit = self.output_path
            self.finished_signal.emit(main_output_path_to_emit, gif_output_path_to_emit, None)

        except FileNotFoundError as fnf_error:
            logger.error(f"File not found during style transfer: {fnf_error}", exc_info=True)
            error_message_to_emit = f"File not found: {str(fnf_error)}"
            self.finished_signal.emit(None, None, error_message_to_emit)
        except ValueError as val_error:
            logger.error(f"Value error during style transfer: {val_error}", exc_info=True)
            error_message_to_emit = f"Input error: {str(val_error)}"
            self.finished_signal.emit(None, None, error_message_to_emit)
        except Exception as e:
            logger.error(f"Style transfer thread failed unexpectedly: {type(e).__name__} - {e}", exc_info=True)
            error_message_to_emit = f"An unexpected error occurred: {type(e).__name__} - {e}"
            self.finished_signal.emit(None, None, error_message_to_emit)
        finally:
            if self.params.get("device", "").startswith("cuda"):
                 logger.debug("Attempting to clear GPU memory from thread.")
                 clear_gpu_memory()