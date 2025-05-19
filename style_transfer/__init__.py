"""
Neural Style Transfer using AdaIN.

This package provides a modern implementation of neural style transfer
using Adaptive Instance Normalization (AdaIN) with PyTorch.
"""

from .models.adain import StyleTransferModel
from .utils.image_processing import ImageProcessor
from .utils.device_utils import get_device, clear_gpu_memory

__version__ = "0.1.0"
APP_NAME = "StyleTransferApp"

__all__ = ['StyleTransferModel', 'ImageProcessor', 'get_device', 'clear_gpu_memory']

logger_name = "style_transfer"
import logging
logger = logging.getLogger(logger_name)
# Basic logging configuration for the package if not handled by application entry point
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# if not logger.handlers:
#     logger.addHandler(handler)
# logger.setLevel(logging.INFO) # Default level for the package logger