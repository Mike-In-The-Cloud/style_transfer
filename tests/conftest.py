import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from style_transfer.models.adain import StyleTransferModel, AdaIN
from style_transfer.utils.image_processing import ImageProcessor
from style_transfer.utils.device_utils import get_device

@pytest.fixture
def device():
    """Fixture to provide the appropriate device for testing."""
    return get_device()

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    # Create a 100x100 RGB image with a gradient
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        img[i, :, 0] = i  # Red channel gradient
        img[:, i, 1] = i  # Green channel gradient
    return Image.fromarray(img)

@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(1, 3, 64, 64)

@pytest.fixture
def model(device):
    """Create a style transfer model instance."""
    model = StyleTransferModel().to(device)
    model.eval()  # Set to evaluation mode
    return model

@pytest.fixture
def processor():
    """Create an image processor instance."""
    return ImageProcessor(max_size=256)

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    return tmp_path