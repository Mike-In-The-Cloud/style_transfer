# Style Transfer Test Suite

This directory contains the automated tests for the Neural Style Transfer project.
We use `pytest` as the test runner and framework, along with several `pytest` plugins
for enhanced functionality such as coverage reporting (`pytest-cov`), parallel execution
(`pytest-xdist`), mocking (`pytest-mock`), and timeouts (`pytest-timeout`).

## Testing Strategy

The tests are organized into modules that generally correspond to the modules in the
main `style_transfer` package. The strategy includes:

-   **Unit Tests:** Focusing on individual functions, methods, and classes to ensure they
    behave as expected in isolation. This includes testing various input conditions,
    edge cases, and error handling.
-   **Integration Tests:** Some tests, particularly in `test_cli.py`, implicitly act as
    integration tests by running the command-line interface and verifying its end-to-end
    behavior, which involves interaction between multiple components (argument parsing,
    image processing, model execution, device management).
-   **Coverage:** We aim for 100% test coverage to ensure all parts of the codebase
    are exercised by the tests. This is checked using `pytest-cov`.

## Test Modules

-   `conftest.py`:
    -   Contains shared `pytest` fixtures used across multiple test files.
    -   For example, it provides a `sample_image` fixture that generates a simple PIL Image
        object, useful for tests requiring image inputs without relying on external files.
    -   It also includes hooks for logging configuration during tests.

-   `test_adain.py`:
    -   **Purpose:** Tests the core AdaIN style transfer model components defined in
        `style_transfer/models/adain.py`.
    -   **Key Tests:**
        -   `AdaIN` layer: Verifies correct normalization and statistics transfer.
        -   `Encoder`: Checks feature extraction output shape and frozen weights.
        -   `Decoder`: Validates output shape and reconstruction capabilities.
        -   `StyleTransferModel` (forward pass): Ensures the end-to-end style transfer
            process produces an output of the correct shape.
        -   `StyleTransferModel` (transfer_style): Tests both standard style transfer and
            the color preservation mode.
        -   Model consistency and gradient flow (if applicable for training later).

-   `test_cli.py`:
    -   **Purpose:** Tests the command-line interface functionality provided by
        `style_transfer/cli.py`.
    -   **Key Tests:**
        -   Basic style transfer execution: Verifies that the CLI can run a style transfer
            and produce an output image.
        -   Specific CLI arguments: Tests options like `--preserve-color`, `--alpha`,
            `--max-size`, and `--device`.
        -   Error handling: Checks for correct behavior with invalid inputs, such as
            missing content/style images, out-of-range alpha values, or invalid device names.
        -   Direct unit tests for helper functions within `cli.py` like `restricted_float`.

-   `test_device_utils.py`:
    -   **Purpose:** Tests the device management utilities in
        `style_transfer/utils/device_utils.py`.
    -   **Key Tests:**
        -   `get_device()`: Verifies correct device selection (CUDA, CPU, specific requests,
            and CUDA unavailable scenarios using mocking).
        -   `clear_gpu_memory()`: Checks that the function runs and attempts to clear memory
            (difficult to assert exact memory clearance, but checks for no errors and proper
            mocking of CUDA calls when unavailable).
        -   `get_optimal_batch_size()`: Tests the batch size estimation logic for both CPU
            and GPU, including edge cases and different model/memory scenarios.

-   `test_image_processing.py`:
    -   **Purpose:** Tests the `ImageProcessor` class from
        `style_transfer/utils/image_processing.py`.
    -   **Key Tests:**
        -   Initialization: Checks `ImageProcessor` instantiation.
        -   `preprocess_image()`: Thoroughly tests image loading and preprocessing with various
            input types (paths, PIL Images, diverse NumPy arrays - different dtypes, shapes,
            channel configurations like RGB, RGBA, grayscale) and error conditions (file not found,
            corrupt images, invalid NumPy arrays, zero-dimension images, tiny images).
        -   `postprocess_image()`: Verifies tensor to PIL Image conversion and normalization reversal.
        -   `batch_preprocess()`: Ensures correct batching of preprocessed images.
        -   `get_image_size()`: Tests image size retrieval for different input types.
        -   Normalization correctness and color space conversions.

## Running Tests

Refer to the "Development" section in the main project `README.md` for instructions
on installing test dependencies and running the test suite with coverage reports.