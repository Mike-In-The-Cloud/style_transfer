import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from style_transfer.utils.image_processing import ImageProcessor
import logging

def test_image_processor_initialization():
    """Test ImageProcessor initialization with different max sizes."""
    for max_size in [256, 512, 1024]:
        processor = ImageProcessor(max_size=max_size)
        assert processor.max_size == max_size

def test_preprocess_image(processor, sample_image, temp_dir):
    """Test image preprocessing with different input types."""
    # Test with PIL Image
    tensor = processor.preprocess_image(sample_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[0] == 1  # Batch dimension
    assert tensor.shape[1] == 3  # RGB channels
    assert tensor.shape[2] <= processor.max_size
    assert tensor.shape[3] <= processor.max_size

    # Test with numpy array
    np_image = np.array(sample_image)
    tensor = processor.preprocess_image(np_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[1] == 3

    # Test with file path
    image_path = temp_dir / "test_image.png"
    sample_image.save(image_path)
    tensor = processor.preprocess_image(str(image_path))
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[1] == 3

def test_postprocess_image(processor, device):
    """Test image postprocessing."""
    # Create a sample tensor
    tensor = torch.randn(1, 3, 256, 256).to(device)

    # Test postprocessing
    image = processor.postprocess_image(tensor)
    assert isinstance(image, Image.Image)
    assert image.mode == 'RGB'

    # Test value range
    np_image = np.array(image)
    assert np_image.min() >= 0
    assert np_image.max() <= 255

def test_batch_preprocess(processor, temp_dir):
    """Test batch preprocessing of multiple images."""
    # Create multiple test images
    images = []
    for i in range(3):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        path = temp_dir / f"test_image_{i}.png"
        img.save(path)
        images.append(path)

    # Test batch preprocessing
    batch = processor.batch_preprocess(images)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape[0] == len(images)
    assert batch.shape[1] == 3

def test_image_size_preservation(processor):
    """Test that image aspect ratio is preserved during preprocessing."""
    # Create a rectangular image
    img = Image.fromarray(np.zeros((200, 100, 3), dtype=np.uint8))

    # Preprocess
    tensor = processor.preprocess_image(img)

    # Check aspect ratio
    original_ratio = 200 / 100
    processed_ratio = tensor.shape[2] / tensor.shape[3]
    assert abs(original_ratio - processed_ratio) < 0.01

def test_normalization(processor):
    """Test that image normalization is applied correctly."""
    # Create a white image (all 255s)
    img_array = np.ones((100, 100, 3), dtype=np.uint8) * 255
    img = Image.fromarray(img_array)

    # Preprocess
    tensor = processor.preprocess_image(img)

    # VGG normalization values
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # Check that values are within a general expected range for normalized images
    # (e.g., typically within ~[-3, 3] for VGG stats)
    assert torch.all(tensor >= -3.0) and torch.all(tensor <= 3.0)

    # For a white image (pixel value 1.0 after ToTensor, before Normalize):
    # Calculate expected normalized values for each channel
    # tensor is (1, C, H, W)
    # expected_r = (1.0 - 0.485) / 0.229  # approx 2.2489
    # expected_g = (1.0 - 0.456) / 0.224  # approx 2.4286
    # expected_b = (1.0 - 0.406) / 0.225  # approx 2.6400

    # More precise check: compare actual tensor values to expected values for white image
    # Create a 1x1 white pixel tensor, normalize it, and compare with a slice of the processed image tensor.
    # This avoids issues with differing H, W during tests if processor.max_size changes.
    white_pixel_pil = Image.fromarray(np.ones((1,1,3), dtype=np.uint8) * 255)
    expected_normalized_pixel = processor.transform(white_pixel_pil) # (3,1,1)

    # Check a sample pixel (e.g., center) from each channel of the processed tensor
    # The processed tensor is (1,3,H,W). All pixels in a channel should be the same for a white image.
    # Taking the mean per channel should give the normalized value for that channel.
    mean_per_channel = tensor.mean(dim=[0,2,3]) # Shape (3)
    assert torch.allclose(mean_per_channel, expected_normalized_pixel.squeeze(), atol=1e-4), \
        f"Normalized values for white image are incorrect. Got {mean_per_channel}, expected ~{expected_normalized_pixel.squeeze()}"

    # The previous assertions for overall mean and std are not valid for a synthetic white image:
    # assert abs(tensor.mean().item()) < 0.1 # Incorrect for white image
    # assert abs(tensor.std().item() - 1.0) < 0.1 # Incorrect for white image

def test_error_handling(processor):
    """Test error handling for invalid inputs."""
    with pytest.raises(Exception):
        processor.preprocess_image(None)

    with pytest.raises(Exception):
        processor.preprocess_image("nonexistent_image.jpg")

    with pytest.raises(Exception):
        processor.preprocess_image(np.zeros((100, 100)))  # 2D array instead of 3D

def test_preprocess_image_generic_pil_error(processor, mocker, tmp_path):
    """Test preprocess_image when PIL.Image.open raises a non-FileNotFound error."""
    mock_image_path = tmp_path / "faulty.png"
    mock_image_path.touch() # Create an empty file

    mocker.patch("PIL.Image.open", side_effect=Exception("PIL generic error"))
    logger_error_spy = mocker.spy(logging.getLogger('style_transfer.utils.image_processing'), 'error')

    with pytest.raises(Exception, match="PIL generic error"):
        processor.preprocess_image(str(mock_image_path))
    logger_error_spy.assert_any_call(f"Error opening image {mock_image_path}: PIL generic error")

def test_preprocess_numpy_float_in_01_range(processor, mocker):
    """Test preprocessing a float numpy array with values in [0,1] range."""
    # This should hit the first if: np.issubdtype(image_rgb.dtype, np.floating) and image_rgb.max() <= 1.0 and image_rgb.min() >= 0.0:
    float_array = np.random.rand(50, 50, 3).astype(np.float32) # Values are already 0-1

    assert float_array.dtype != np.uint8
    assert np.issubdtype(float_array.dtype, np.floating)
    assert float_array.min() >= 0.0
    assert float_array.max() <= 1.0

    logger_warning_spy = mocker.spy(logging.getLogger('style_transfer.utils.image_processing'), 'warning')

    tensor = processor.preprocess_image(float_array)
    assert tensor.shape[1] == 3
    logger_warning_spy.assert_not_called() # No warning should be logged for this case

def test_preprocess_numpy_grayscale_hw1(processor, mocker):
    """Test preprocessing a HxWx1 grayscale numpy array."""
    logger_warning_spy = mocker.spy(logging.getLogger('style_transfer.utils.image_processing'), 'warning')
    gray_array = np.random.randint(0, 255, (100, 100, 1), dtype=np.uint8)
    tensor = processor.preprocess_image(gray_array)
    assert tensor.shape == (1, 3, processor.max_size, processor.max_size) # or other expected scaled size
    logger_warning_spy.assert_any_call("Input numpy array is 3D single-channel (grayscale). Converting to RGB by duplicating channels.")

def test_preprocess_numpy_rgba(processor, mocker):
    """Test preprocessing an RGBA HxWx4 numpy array."""
    logger_info_spy = mocker.spy(logging.getLogger('style_transfer.utils.image_processing'), 'info')
    rgba_array = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
    tensor = processor.preprocess_image(rgba_array)
    assert tensor.shape == (1, 3, processor.max_size, processor.max_size)
    logger_info_spy.assert_any_call("Input numpy array is RGBA. Converting to RGB.")

def test_preprocess_numpy_invalid_channels(processor):
    """Test preprocessing numpy array with invalid number of channels."""
    invalid_channel_array = np.random.randint(0, 255, (100, 100, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported numpy array channel size: 5"):
        processor.preprocess_image(invalid_channel_array)

def test_preprocess_numpy_invalid_dims(processor):
    """Test preprocessing numpy array with invalid dimensions."""
    invalid_dims_array = np.random.randint(0, 255, (100, 100, 3, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported numpy array dimensions: 4"):
        processor.preprocess_image(invalid_dims_array)

def test_preprocess_numpy_int_in_range(processor, mocker):
    """Test preprocessing integer numpy array (e.g. np.short) already in uint8 value range."""
    # Using np.short (typically int16) to be very explicit with a non-uint8 integer type.
    # Values are clearly within [0, 255].
    # This is designed to hit: elif np.issubdtype(image_rgb.dtype, np.integer) and image_rgb.min() >=0 and image_rgb.max() <=255:
    int_values = [[[10, 20, 30], [200, 210, 220]],
                  [[40, 50, 60], [230, 240, 250]]]
    int_array = np.array(int_values, dtype=np.short) # np.short is usually np.int16

    # Verify conditions for the target branch are met by this input
    assert int_array.dtype != np.uint8
    assert np.issubdtype(int_array.dtype, np.integer)
    assert int_array.min() >= 0
    assert int_array.max() <= 255

    logger_warning_spy = mocker.spy(logging.getLogger('style_transfer.utils.image_processing'), 'warning')

    # Preprocess the image
    tensor = processor.preprocess_image(int_array)

    # Basic check on output
    assert tensor.shape[1] == 3 # Check that it was processed to 3 channels

    # Crucially, ensure no warning was logged.
    # This implies the target 'elif' branch was taken (and its astype conversion happened),
    # and the final 'else' branch (which logs a warning and clips) was not.
    logger_warning_spy.assert_not_called()

def test_preprocess_numpy_other_dtype_clip(processor, mocker):
    """Test preprocessing numpy array of other dtypes that need clipping."""
    logger_warning_spy = mocker.spy(logging.getLogger('style_transfer.utils.image_processing'), 'warning')
    # float64 array not in [0,1] range, to trigger clipping and warning
    float_array = np.random.rand(100, 100, 3).astype(np.float64) * 500 - 100 # Values outside 0-255 after potential scaling
    tensor = processor.preprocess_image(float_array)
    assert tensor.shape[1] == 3
    logger_warning_spy.assert_any_call(mocker.ANY) # Check that a warning was logged
    # A more specific check for the log message content could be added here if needed

def test_preprocess_unsupported_type(processor):
    """Test preprocessing with an unsupported input type."""
    with pytest.raises(TypeError, match="Unsupported image type: <class 'list'>"):
        processor.preprocess_image([1,2,3])

def test_preprocess_non_rgb_pil_image(processor, mocker):
    """Test preprocessing a non-RGB PIL image (e.g., Grayscale 'L', or 'RGBA')."""
    logger_info_spy = mocker.spy(logging.getLogger('style_transfer.utils.image_processing'), 'info')
    # Test with Grayscale
    gray_pil = Image.new('L', (100,100), color='gray')
    tensor_gray = processor.preprocess_image(gray_pil)
    assert tensor_gray.shape[1] == 3
    logger_info_spy.assert_any_call("Image mode is L. Converting to RGB.")
    logger_info_spy.reset_mock() # Reset for next call
    # Test with RGBA
    rgba_pil = Image.new('RGBA', (100,100), color=(10,20,30,40))
    tensor_rgba = processor.preprocess_image(rgba_pil)
    assert tensor_rgba.shape[1] == 3
    logger_info_spy.assert_any_call("Image mode is RGBA. Converting to RGB.")

def test_preprocess_zero_dimension_image(processor):
    """Test preprocessing a zero-dimension image."""
    # PIL might not allow creating a 0x0 image directly easily, let's try 0xN or Nx0
    zero_dim_img_w0 = Image.new('RGB', (0, 10))
    with pytest.raises(ValueError, match="Image dimensions are zero or invalid"):
        processor.preprocess_image(zero_dim_img_w0)

    zero_dim_img_h0 = Image.new('RGB', (10, 0))
    with pytest.raises(ValueError, match="Image dimensions are zero or invalid"):
        processor.preprocess_image(zero_dim_img_h0)

def test_preprocess_tiny_image_scaling(processor):
    """Test preprocessing a very small image that might scale to zero without care."""
    # This test covers the case where int(w*scale) or int(h*scale) becomes 0
    tiny_img_one_pixel_wide = Image.new('RGB', (1, 200)) # Will scale width to 0 if max_size is e.g. 128 and not handled
    processor_small_max = ImageProcessor(max_size=128)
    tensor = processor_small_max.preprocess_image(tiny_img_one_pixel_wide)
    assert tensor.shape[2] > 0 and tensor.shape[3] > 0 # Height and Width > 0

    tiny_img_one_pixel_high = Image.new('RGB', (200, 1))
    tensor = processor_small_max.preprocess_image(tiny_img_one_pixel_high)
    assert tensor.shape[2] > 0 and tensor.shape[3] > 0 # Height and Width > 0

# Tests for get_image_size
def test_get_image_size_various_inputs(processor, tmp_path, sample_image):
    """Test get_image_size with string path, Path object, numpy array, and PIL image."""
    original_pil_image = sample_image
    w, h = original_pil_image.size

    # Test with PIL Image
    assert processor.get_image_size(original_pil_image) == (w,h)

    # Test with string path
    img_path_str = str(tmp_path / "get_size_test.png")
    original_pil_image.save(img_path_str)
    assert processor.get_image_size(img_path_str) == (w,h)

    # Test with Path object
    img_path_obj = tmp_path / "get_size_test_pathlib.png"
    original_pil_image.save(img_path_obj)
    assert processor.get_image_size(img_path_obj) == (w,h)

    # Test with numpy array
    np_array = np.array(original_pil_image)
    assert processor.get_image_size(np_array) == (w,h)