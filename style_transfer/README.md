# Style Transfer Core (`style_transfer`)

This directory is the heart of the neural style transfer application, containing all the core logic for both command-line (CLI) and graphical user interface (GUI) operations, model definitions, and utility functions.

## Directory Structure

```text
style_transfer/
├── __init__.py                 # Package initialisation and exports
├── cli.py                      # Command-Line Interface logic and main processing pipeline
├── gui.py                      # Main Graphical User Interface window
├── models/                     # PyTorch model definitions
│   ├── __init__.py
│   ├── adain.py
│   └── johnson_transformer.py
├── gui_components/             # Reusable GUI widgets and the style transfer worker thread
│   ├── __init__.py
│   ├── adain_options_widget.py
│   ├── johnson_advanced_options_widget.py
│   ├── johnson_model_selection_widget.py
│   └── style_transfer_thread.py
└── utils/                      # Utility modules (device, image, model, video)
    ├── __init__.py
    ├── device_utils.py
    ├── image_processing.py
    ├── model_utils.py
    └── video_utils.py
```

## Core Files

### `__init__.py`

- Initialises the `style_transfer` package.
- Exports key classes and functions like `StyleTransferModel`, `ImageProcessor`, `get_device`, and `clear_gpu_memory` for easier access from outside the package.
- Defines package-level constants like `__version__` and `APP_NAME`.
- Sets up a basic logger for the package.

### `cli.py`

- **Command-Line Interface (CLI) Entry Point:** Implements the CLI using `argparse` to accept various parameters for style transfer.
- **`run_style_transfer_pipeline(...)` function:** This is the central processing function for the entire application (used by both CLI and GUI). It handles:
  - Device selection (CPU/GPU).
  - Loading content and style images.
  - Conditional logic for AdaIN and Johnson methods.
  - **AdaIN Pathway:**
    - Initialises the `StyleTransferModel` (from `models.adain`) with VGG and decoder weights.
    - Performs style transfer using `StyleTransferModel.transfer_style()`.
    - Optionally generates a sequence of frames for an alpha-blending GIF using `StyleTransferModel.generate_alpha_sequence_frames()`.
  - **Johnson Pathway:**
    - Loads pre-trained Johnson `TransformerNetwork` weights using `utils.model_utils.load_johnson_model()`.
    - Wraps the loaded network in a `JohnsonNet` instance (from `models.johnson_transformer`).
    - Performs style transfer by calling the `JohnsonNet` instance.
    - Optionally blends the output with the original content image.
    - Optionally generates a sequence of frames for a style-interpolation GIF using `utils.video_utils.generate_interpolated_frames()`.
  - Saving the final stylised image.
  - Saving generated GIFs using `utils.video_utils.create_gif_from_frames()`.
  - Error handling and logging.
- **`main()` function:** Parses CLI arguments and calls `run_style_transfer_pipeline`.

### `gui.py`

- **Main GUI Window (`MainWindow` class):** Implements the main application window using PySide6.
- **User Interface Layout:**
  - Displays content, style (for AdaIN), and output images, plus a preview for generated GIFs.
  - Provides file selection buttons for content and style images.
  - Includes controls for selecting the style transfer method (AdaIN/Johnson) and computation device.
- **Integration of GUI Components:**
  - Uses `AdaINOptionsWidget` for AdaIN-specific parameters (alpha, colour preservation, GIF settings).
  - Uses `JohnsonModelSelectionWidget` for selecting pre-trained or custom Johnson models.
  - Uses `JohnsonAdvancedOptionsWidget` for Johnson-specific parameters (output blend, GIF settings).
- **Interaction Logic:**
  - Dynamically updates UI controls based on the selected method.
  - Handles user interactions like button clicks and value changes.
- **Style Transfer Execution:**
  - Instantiates and runs `StyleTransferThread` (from `gui_components`) to perform the style transfer in a background thread, preventing the GUI from freezing.
  - Passes all necessary parameters gathered from the UI to the `StyleTransferThread`.
- **Displaying Results:**
  - Updates image and GIF display areas upon completion of the style transfer.
  - Manages a progress bar during processing.
  - Provides options to save the generated image and GIF.
- **`main()` function:** Initialises and runs the Qt application and `MainWindow`.

## Subdirectories

### `models/`

This subdirectory contains the PyTorch model definitions.

- **`__init__.py`:** Makes the `models` directory a Python package and exports the core model classes (`AdaIN`, `Encoder`, `Decoder`, `StyleTransferModel`, `TransformerNetwork`, `TransformerNetworkTanh`, `JohnsonNet`).
- **`adain.py`:**
  - **`Encoder` class:** VGG19-based encoder, loads pre-trained `vgg_normalised.pth` weights.
  - **`Decoder` class:** Reflect-padded convolutional decoder, loads pre-trained `decoder.pth` weights.
  - **`AdaIN` class:** Implements the Adaptive Instance Normalisation layer.
  - **`StyleTransferModel` class:**
    - Combines the `Encoder`, `AdaIN` layer, and `Decoder`.
    - `transfer_style()`: Core method to perform AdaIN style transfer.
    - `generate_alpha_sequence_frames()`: Generates frames for an alpha-blending GIF, showing the transition from content to stylised image.
- **`johnson_transformer.py`:**
  - **`TransformerNetwork` & `TransformerNetworkTanh` classes:** Define the architectures for Johnson's Fast Neural Style Transfer (Perceptual Losses) model. These networks are pre-trained for specific styles.
  - **`JohnsonNet` class:** A wrapper around `TransformerNetwork` or `TransformerNetworkTanh` to provide a consistent interface for the stylisation pipeline.

### `gui_components/`

This subdirectory contains reusable Qt widgets for the GUI and the style transfer worker thread.

- **`__init__.py`:** Makes `gui_components` a Python package.
- **`adain_options_widget.py` (`AdaINOptionsWidget` class):**
  - Provides GUI controls specific to the AdaIN method:
    - Alpha (style strength) slider/spinbox.
    - "Preserve Colour" checkbox.
    - GIF generation checkbox.
    - GIF frame count and duration spinboxes.
    - GIF "Ping-Pong" effect checkbox.
- **`johnson_model_selection_widget.py` (`JohnsonModelSelectionWidget` class):**
  - Provides GUI controls for selecting the Johnson model:
    - Dropdown to select from pre-trained models found in the `models/` (root) directory.
    - Button to load a custom Johnson model (`.pth` file).
- **`johnson_advanced_options_widget.py` (`JohnsonAdvancedOptionsWidget` class):**
  - Provides GUI controls for advanced Johnson method options:
    - Output blend alpha slider/spinbox.
    - GIF generation checkbox.
    - GIF frame count, duration, and style intensity spinboxes.
    - GIF "Ping-Pong" effect checkbox.
- **`style_transfer_thread.py` (`StyleTransferThread` class):**
  - A `QThread` subclass responsible for running the style transfer process in the background.
  - Takes parameters from the `MainWindow` (image paths, selected method, specific options).
  - **Crucially, it calls the `run_style_transfer_pipeline()` function from `cli.py` to perform the actual work.**
  - Emits signals (`finished_signal`) upon completion or error, passing back output paths or error messages to the `MainWindow`.
  - Includes logic to clear GPU memory after processing if CUDA was used.

### `utils/`

This subdirectory contains various utility modules.

- **`__init__.py`:** Makes `utils` a Python package and exports key utility functions (`ImageProcessor`, `get_device`, `clear_gpu_memory`, `get_optimal_batch_size`).
- **`device_utils.py`:**
  - `get_device()`: Determines the appropriate PyTorch device (CUDA, MPS, CPU) based on availability and user preference.
  - `get_available_devices()`: Returns a list of available computation devices.
  - `clear_gpu_memory()`: Utility to explicitly clear CUDA GPU memory.
  - `get_optimal_batch_size()`: Utility for heuristically estimating an optimal batch size for GPU processing based on available memory.
- **`image_processing.py`:**
  - `load_image_as_tensor()`: Loads an image from a file path, resizes it, normalises it, and converts it to a PyTorch tensor. Handles different normalisation based on model type (AdaIN expects 0-1, Johnson often uses 0-255).
  - `save_tensor_as_image()`: Converts a PyTorch tensor back to an image and saves it to a file. Handles denormalisation.
  - `get_image_size()`: Retrieves dimensions of an image file.
  - `ImageProcessor` class: (If present and used) May encapsulate some of these image operations. *Note: The `style_transfer.utils/__init__.py` exports `ImageProcessor`, suggesting its presence and use.*
- **`model_utils.py`:**
  - `load_johnson_model()`: Loads a pre-trained Johnson `TransformerNetwork` from a `.pth` file and prepares it for use (e.g., moves to device).
- **`video_utils.py`:**
  - `create_gif_from_frames()`: Takes a directory of image frames and creates an animated GIF using `imageio`. Supports a "ping-pong" effect (plays forwards then backwards).
  - `generate_interpolated_frames()`: Generates frames for Johnson model GIFs by interpolating between the content image and the fully stylised image, or by varying style intensity.