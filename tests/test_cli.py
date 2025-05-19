import pytest
import subprocess
from pathlib import Path
from PIL import Image
import os

# Path to the parent directory of the 'style_transfer' package
PROJECT_ROOT = Path(__file__).parent.parent

def run_cli_command(args_list):
    """Helper function to run the CLI script with given arguments using module execution."""
    # Command to run: python -m style_transfer.cli [args...]
    # This requires that the tests are run from an environment where 'style_transfer' is in PYTHONPATH
    # or that pytest is run from the project root.
    command = ["python", "-m", "style_transfer.cli"] + args_list
    # We need to set the PYTHONPATH to include the project root so that style_transfer can be found as a module.
    # Alternatively, running pytest from the project root often handles this automatically.
    # For explicit control in subprocess, modify env:
    env = os.environ.copy()
    # Add project root to PYTHONPATH if not already there or if needed for subprocess context
    current_python_path = env.get("PYTHONPATH", "")
    if str(PROJECT_ROOT) not in current_python_path.split(os.pathsep):
        env["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{current_python_path}"

    return subprocess.run(command, capture_output=True, text=True, check=False, env=env, cwd=PROJECT_ROOT)

def test_cli_basic_style_transfer(tmp_path, sample_image):
    """Test basic style transfer via CLI, ensuring an output image is created."""
    content_img_path = tmp_path / "content.png"
    style_img_path = tmp_path / "style.png"
    output_img_path = tmp_path / "output.png"

    # Create dummy content and style images for the test
    sample_image.save(content_img_path)
    # For simplicity, using the same image for style, or create a different one if needed
    sample_image.rotate(90).save(style_img_path)

    args = [
        "--content", str(content_img_path),
        "--style", str(style_img_path),
        "--output", str(output_img_path),
        "--max-size", "128"  # Use a small size for faster testing
    ]

    result = run_cli_command(args)

    print("CLI STDOUT:", result.stdout)
    print("CLI STDERR:", result.stderr)

    assert result.returncode == 0, f"CLI script failed with error: {result.stderr}"
    assert output_img_path.exists(), "Output image was not created."

    # Optionally, check if the output image is valid
    try:
        with Image.open(output_img_path) as img:
            assert img.format is not None
            assert img.size[0] > 0 and img.size[1] > 0
    except Exception as e:
        pytest.fail(f"Output image is invalid or corrupted: {e}")

def test_cli_color_preservation(tmp_path, sample_image):
    """Test style transfer with color preservation via CLI."""
    content_img_path = tmp_path / "content_color.png"
    style_img_path = tmp_path / "style_color.png"
    output_img_path = tmp_path / "output_color.png"

    sample_image.save(content_img_path)
    sample_image.rotate(45).save(style_img_path)

    args = [
        "--content", str(content_img_path),
        "--style", str(style_img_path),
        "--output", str(output_img_path),
        "--preserve-color",
        "--max-size", "128"
    ]

    result = run_cli_command(args)

    print("CLI STDOUT:", result.stdout)
    print("CLI STDERR:", result.stderr)

    assert result.returncode == 0, f"CLI script failed with error: {result.stderr}"
    assert output_img_path.exists(), "Output image with color preservation was not created."

    try:
        with Image.open(output_img_path) as img:
            assert img.format is not None
    except Exception as e:
        pytest.fail(f"Output image (color preserved) is invalid: {e}")


def test_cli_missing_content_image(tmp_path):
    """Test CLI behavior when content image is missing."""
    style_img_path = tmp_path / "style_missing.png" # Needs to exist for arg parsing
    Image.new('RGB', (60, 30), color = 'red').save(style_img_path)
    output_img_path = tmp_path / "output_missing_content.png"

    args = [
        "--content", "non_existent_content.png",
        "--style", str(style_img_path),
        "--output", str(output_img_path)
    ]

    result = run_cli_command(args)

    # Expecting a non-zero return code because the main function in cli.py re-raises exceptions
    assert result.returncode != 0, "CLI should fail when content image is missing."
    # Check for specific error related to file not found in stderr
    # This depends on how cli.py logs/handles FileNotFoundError
    assert "FileNotFoundError" in result.stderr or "Error opening image" in result.stderr or "No such file or directory" in result.stderr


def test_cli_missing_style_image(tmp_path, sample_image):
    """Test CLI behavior when style image is missing."""
    content_img_path = tmp_path / "content_missing_style.png" # Needs to exist
    sample_image.save(content_img_path)
    output_img_path = tmp_path / "output_missing_style.png"

    args = [
        "--content", str(content_img_path),
        "--style", "non_existent_style.png",
        "--output", str(output_img_path)
    ]

    result = run_cli_command(args)

    assert result.returncode != 0, "CLI should fail when style image is missing."
    assert "FileNotFoundError" in result.stderr or "Style image not found" in result.stderr


def test_cli_invalid_alpha(tmp_path, sample_image):
    """Test CLI behavior with an invalid alpha value."""
    content_img_path = tmp_path / "content_alpha.png"
    style_img_path = tmp_path / "style_alpha.png"
    output_img_path = tmp_path / "output_alpha.png"

    sample_image.save(content_img_path)
    sample_image.save(style_img_path)

    args = [
        "--content", str(content_img_path),
        "--style", str(style_img_path),
        "--output", str(output_img_path),
        "--alpha", "2.0"  # Invalid alpha
    ]

    result = run_cli_command(args)

    # The model's transfer_style might clamp alpha, or argparse might not validate range unless specified.
    # For now, let's assume the script might run but produce an image based on clamped alpha (1.0).
    # A more robust test would involve checking the style strength in the output image if possible,
    # or modifying argparse to have choices/range for alpha.
    # The current AdaiN model internally clamps alpha to [0,1] effectively, so script should run.
    # However, a UserWarning could be nice, or argparse level validation.
    # The current Adain implementation doesn't validate alpha range before use, it just uses it.
    # Let's assume for now the script will succeed. If the model had validation and raised error, this would fail.
    # The `transfer_style` method does not validate alpha range explicitly before StyleTransferModel.forward
    # And StyleTransferModel.forward does not validate alpha either. It will just be used.
    # So, the script will actually run and produce an output.
    # Let's refine this. Ideally, our model or CLI should warn/error for alpha > 1.
    # For now, let's assume the model clamps it and the CLI succeeds.
    # If the model raised an error for alpha > 1, this test would need to assert result.returncode != 0

    # Current main() in cli.py re-raises exceptions.
    # The StyleTransferModel.forward does not raise error for alpha > 1.
    # Let's check if argparse itself fails for bad float, it does not for "2.0"
    # The current model will simply use alpha=2.0.
    # For this test to be meaningful for "invalid alpha", we'd need the model/CLI to *define* what's invalid.
    # Let's assume for now that our requirement for "invalid alpha" means argparse failing.
    # Argparse will fail if type=float gets a non-float string.
    # If alpha must be 0-1, this validation should be in the model or argparse.

    args_bad_type = [
        "--content", str(content_img_path),
        "--style", str(style_img_path),
        "--output", str(output_img_path),
        "--alpha", "not_a_float"
    ]
    result_bad_type = run_cli_command(args_bad_type)
    assert result_bad_type.returncode != 0, "CLI should fail for alpha value that cannot be parsed as float."
    assert "argument --alpha: 'not_a_float' not a floating-point literal" in result_bad_type.stderr

def test_cli_alpha_out_of_range(tmp_path, sample_image):
    """Test CLI behavior with alpha value out of the [0.0, 1.0] range."""
    content_img_path = tmp_path / "content_alpha_range.png"
    style_img_path = tmp_path / "style_alpha_range.png"
    output_img_path = tmp_path / "output_alpha_range.png"

    sample_image.save(content_img_path)
    sample_image.save(style_img_path)

    args_too_high = [
        "--content", str(content_img_path),
        "--style", str(style_img_path),
        "--output", str(output_img_path),
        "--alpha", "1.5"
    ]
    result_too_high = run_cli_command(args_too_high)
    assert result_too_high.returncode != 0, "CLI should fail for alpha > 1.0"
    assert "not in range [0.0, 1.0]" in result_too_high.stderr

    args_too_low = [
        "--content", str(content_img_path),
        "--style", str(style_img_path),
        "--output", str(output_img_path),
        "--alpha", "-0.5"
    ]
    result_too_low = run_cli_command(args_too_low)
    assert result_too_low.returncode != 0, "CLI should fail for alpha < 0.0"
    assert "not in range [0.0, 1.0]" in result_too_low.stderr

def test_cli_invalid_device(tmp_path, sample_image):
    """Test CLI behavior with an invalid device argument."""
    content_img_path = tmp_path / "content_device.png"
    style_img_path = tmp_path / "style_device.png"
    output_img_path = tmp_path / "output_device.png"

    sample_image.save(content_img_path)
    sample_image.save(style_img_path)

    args = [
        "--content", str(content_img_path),
        "--style", str(style_img_path),
        "--output", str(output_img_path),
        "--device", "mps"  # Invalid device choice
    ]

    result = run_cli_command(args)
    assert result.returncode != 0, "CLI should fail for invalid device choice."
    # argparse error messages for choices usually include something like "invalid choice: 'mps' (choose from 'cuda', 'cpu')"
    assert "invalid choice" in result.stderr
    assert "'mps'" in result.stderr # Check that the invalid choice is mentioned
    assert "'cuda', 'cpu'" in result.stderr # Check that available choices are mentioned

# Direct unit tests for restricted_float
from style_transfer.cli import restricted_float
# pytest and ArgumentTypeError are already imported if this is in test_cli.py
# but if placing in a new file, they would be needed.
from argparse import ArgumentTypeError # Ensure this is imported

def test_restricted_float_valid():
    """Test restricted_float with valid inputs."""
    assert restricted_float("0.5") == 0.5
    assert restricted_float("0.0") == 0.0
    assert restricted_float("1.0") == 1.0
    # Test with actual float input as well, though argparse provides string
    assert restricted_float(0.7) == 0.7

def test_restricted_float_invalid_type():
    """Test restricted_float with a non-float string."""
    with pytest.raises(ArgumentTypeError, match="not a floating-point literal"):
        restricted_float("not_a_float")

def test_restricted_float_out_of_range():
    """Test restricted_float with out-of-range values."""
    with pytest.raises(ArgumentTypeError, match=r"1.1 not in range \[0.0, 1.0\]"):
        restricted_float("1.1")
    with pytest.raises(ArgumentTypeError, match=r"-0.1 not in range \[0.0, 1.0\]"):
        restricted_float("-0.1")