"""
Image processing utilities for style transfer.

This module provides the ImageProcessor class, which handles loading, preprocessing,
and postprocessing of images. Preprocessing includes resizing, tensor conversion,
and normalization suitable for VGG-based models. Postprocessing converts tensors
back to PIL Images.
"""
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError # Added UnidentifiedImageError for specific exception handling
import numpy as np
from pathlib import Path
from typing import Tuple, Union, List # Tuple unused, can be removed if not used elsewhere
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handles image loading, preprocessing, and postprocessing for style transfer.

    This class standardizes image handling by:
    - Loading images from various sources (paths, PIL objects, NumPy arrays).
    - Resizing images to a maximum dimension while maintaining aspect ratio.
    - Applying VGG-specific normalization (mean and standard deviation) for 'adain' model type.
    - Scaling images to [0, 255] range for 'johnson' model type.
    - Converting images to PyTorch tensors and vice-versa.
    - Providing batch preprocessing capabilities.
    """
    def __init__(self, max_size: int = 512):
        """
        Initializes the ImageProcessor.

        Args:
            max_size (int, optional): The maximum dimension (width or height) to which
                                      images will be resized. Defaults to 512.
        """
        self.max_size = max_size
        # VGG19 normalization parameters (from ImageNet) - used for 'adain' model
        self.mean_values = [0.485, 0.456, 0.406]
        self.std_values = [0.229, 0.224, 0.225]

        # Basic transforms, components will be used conditionally
        self.to_tensor = transforms.ToTensor() # Converts PIL Image (H, W, C) in range [0, 255] to (C, H, W) in range [0.0, 1.0]
        self.to_pil_image = transforms.ToPILImage()

        self.vgg_normalize = transforms.Normalize(mean=self.mean_values, std=self.std_values)
        self.vgg_denormalize = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1/s for s in self.std_values]),
            transforms.Normalize(mean=[-m for m in self.mean_values], std=[1., 1., 1.]),
        ])

    def preprocess_image(self, image_input: Union[str, Image.Image, np.ndarray, Path], model_type: str = 'adain') -> torch.Tensor:
        """
        Loads, resizes, and preprocesses a single image for style transfer.

        The preprocessing involves:
        1. Loading the image from path, Path object, PIL Image, or NumPy array.
        2. Converting NumPy arrays or non-RGB PIL images to RGB PIL Image.
        3. Resizing the image to `self.max_size` on its largest dimension, preserving aspect ratio.
        4. Applying transformations based on model_type.
        5. Adding a batch dimension.

        Args:
            image_input (Union[str, Image.Image, np.ndarray, Path]): The input image.
                Can be a file path (string), a Path object, a PIL Image object,
                or a NumPy array (H, W, C) or (H, W).
            model_type (str, optional): The model type for preprocessing. Defaults to 'adain'.
                'adain': Apply VGG-specific normalization (mean and standard deviation).
                'johnson': Scale image to [0, 255] range.

        Returns:
            torch.Tensor: The preprocessed image tensor of shape (1, 3, H_resized, W_resized).

        Raises:
            FileNotFoundError: If `image_input` is a path and the file is not found.
            UnidentifiedImageError: If `image_input` is a path and PIL cannot open or identify the image.
            ValueError: If NumPy array has unsupported dimensions/channels or image dimensions are invalid.
            TypeError: If `image_input` type is unsupported.
        """
        img: Image.Image # Type hint for img variable

        if isinstance(image_input, (str, Path)):
            try:
                # Ensure path is a string for Image.open
                img = Image.open(str(image_input))
            except FileNotFoundError:
                logger.error(f"Image file not found: {image_input}")
                raise
            except UnidentifiedImageError:
                logger.error(f"Cannot open or identify image file: {image_input}")
                raise # Re-raise to allow specific handling upstream if needed
            except Exception as e: # Catch other PIL errors
                logger.error(f"Error opening image {image_input} with PIL: {e}")
                raise
        elif isinstance(image_input, np.ndarray):
            # Handle NumPy array input
            if image_input.ndim == 2: # Grayscale HxW
                # As per project requirements (test_error_handling), 2D arrays are explicitly not supported
                # for direct style transfer as color information is expected.
                logger.error("Input numpy array is 2D (grayscale). Expected 3D (H, W, C) for style transfer.")
                raise ValueError("Input numpy array is 2D (grayscale). Expected 3D (H, W, C) for style transfer.")
            elif image_input.ndim == 3:
                # Handle 3D NumPy arrays (H, W, C)
                if image_input.shape[2] == 1: # Grayscale HxWx1
                    logger.warning("Input numpy array is 3D single-channel (grayscale). Converting to RGB by duplicating channels.")
                    # Duplicate the single channel to create an RGB image
                    image_rgb = np.concatenate([image_input] * 3, axis=2)
                elif image_input.shape[2] == 4: # RGBA HxWx4
                    logger.info("Input numpy array is RGBA. Converting to RGB by discarding alpha channel.")
                    image_rgb = image_input[..., :3] # Take only RGB channels
                elif image_input.shape[2] == 3: # RGB HxWx3
                    image_rgb = image_input # Already in RGB format
                else:
                    err_msg = f"Unsupported numpy array channel size: {image_input.shape[2]}. Expected 1 (grayscale), 3 (RGB), or 4 (RGBA) channels."
                    logger.error(err_msg)
                    raise ValueError(err_msg)
            else:
                err_msg = f"Unsupported numpy array dimensions: {image_input.ndim}. Expected 3D (H, W, C)."
                logger.error(err_msg)
                raise ValueError(err_msg)

            # Convert NumPy array to PIL Image, handling dtype conversions
            if image_rgb.dtype != np.uint8:
                # If float in [0,1] range, scale to [0,255]
                if np.issubdtype(image_rgb.dtype, np.floating) and image_rgb.max() <= 1.0 and image_rgb.min() >= 0.0:
                    logger.debug(f"NumPy array is float in [0,1] range, scaling to uint8.")
                    image_rgb = (image_rgb * 255).astype(np.uint8)
                # If integer in [0,255] range but not uint8, cast
                elif np.issubdtype(image_rgb.dtype, np.integer) and image_rgb.min() >=0 and image_rgb.max() <=255:
                    logger.debug(f"NumPy array is integer in [0,255] range (dtype {image_rgb.dtype}), casting to uint8.")
                    image_rgb = image_rgb.astype(np.uint8)
                else: # For other dtypes or ranges, clip and cast
                    logger.warning(f"NumPy array has dtype {image_rgb.dtype} and range [{image_rgb.min()},{image_rgb.max()}]. Clipping to [0,255] and casting to uint8.")
                    image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
            img = Image.fromarray(image_rgb, 'RGB') # Explicitly state mode for clarity
        elif isinstance(image_input, Image.Image):
            # Input is already a PIL Image
            img = image_input
        else:
            err_msg = f"Unsupported image input type: {type(image_input)}. Expected string path, Path object, PIL Image, or NumPy array."
            logger.error(err_msg)
            raise TypeError(err_msg)

        # Ensure image is in RGB mode for consistency
        if img.mode != 'RGB':
            logger.info(f"Image mode is {img.mode}. Converting to RGB.")
            img = img.convert('RGB')

        # Resize image while maintaining aspect ratio
        w, h = img.size
        if w == 0 or h == 0:
            err_msg = f"Image dimensions are zero or invalid: ({w}x{h}). Cannot process."
            logger.error(err_msg)
            raise ValueError(err_msg)

        scale = self.max_size / max(w, h)
        # Calculate new dimensions, ensuring they are at least 1x1
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        new_size = (new_w, new_h)

        logger.debug(f"Resizing image from ({w}x{h}) to {new_size} using LANCZOS resampling.")
        img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Apply transformations based on model_type
        tensor = self.to_tensor(img) # Converts to (C, H, W) in range [0.0, 1.0]

        if model_type == 'adain':
            tensor = self.vgg_normalize(tensor)
        elif model_type == 'johnson':
            tensor = tensor * 255.0 # Scale to [0, 255] for Johnson model
        else:
            raise ValueError(f"Unsupported model_type for preprocessing: {model_type}")

        # Add batch dimension to comply with model input shape (1, C, H, W)
        return tensor.unsqueeze(0)

    def postprocess_image(self, tensor: torch.Tensor, model_type: str = 'adain') -> Image.Image:
        """
        Converts a PyTorch tensor back to a PIL Image after style transfer.

        The postprocessing involves:
        1. Removing the batch dimension.
        2. Moving the tensor to CPU.
        3. Clamping values to the [0, 1] range (important if Tanh wasn't used or color preservation altered range).
        4. Applying reverse transformations: un-normalizing and converting to PIL.

        Args:
            tensor (torch.Tensor): Input tensor, typically the output of the style transfer model,
                                   expected shape (1, 3, H, W) or (3, H, W).
            model_type (str, optional): The model type for postprocessing. Defaults to 'adain'.
                'adain': Denormalize for VGG-based models and clamp to [0,1] before converting to PIL
                'johnson': Clamp to [0,255], convert to uint8, then to PIL.

        Returns:
            Image.Image: The postprocessed PIL Image, ready to be saved or displayed.
        """
        # Ensure tensor is on CPU and remove batch dimension if present
        processed_tensor = tensor.cpu().squeeze(0)

        if model_type == 'adain':
            # Denormalize for VGG-based models and clamp to [0,1] before converting to PIL
            processed_tensor = self.vgg_denormalize(processed_tensor)
            processed_tensor = torch.clamp(processed_tensor, 0, 1) # Clamp after denorm
            pil_image = self.to_pil_image(processed_tensor)
        elif model_type == 'johnson':
            # Johnson TransformerNetwork (without Tanh) outputs values likely in [0, 255] range.
            # Clamp to [0, 255], then normalize to [0, 1] for PIL conversion.
            processed_tensor = torch.clamp(processed_tensor, 0, 255)
            processed_tensor = processed_tensor / 255.0
            pil_image = self.to_pil_image(processed_tensor) # Handles float [0,1] tensor
        else:
            raise ValueError(f"Unsupported model_type for postprocessing: {model_type}")

        return pil_image

    def batch_preprocess(self, images: List[Union[str, Image.Image, np.ndarray, Path]], model_type: str = 'adain') -> torch.Tensor:
        """
        Preprocesses a list of images and stacks them into a single batch tensor.

        Args:
            images (List[Union[str, Image.Image, np.ndarray, Path]]): A list of images.
                Each image can be a file path, Path object, PIL Image, or NumPy array.
            model_type (str, optional): The model type for preprocessing. Defaults to 'adain'.
                'adain': Apply VGG-specific normalization (mean and standard deviation).
                'johnson': Scale image to [0, 255] range.

        Returns:
            torch.Tensor: A batch of preprocessed image tensors, stacked along dimension 0.
                          Shape: (N, 3, H_resized, W_resized), where N is the number of images.
        """
        processed_images = [self.preprocess_image(img, model_type=model_type) for img in images]
        return torch.cat(processed_images, dim=0)

    def get_image_size(self, image_input: Union[str, Image.Image, np.ndarray, Path]) -> Tuple[int, int]:
        """
        Retrieves the dimensions (width, height) of an image.

        Args:
            image_input (Union[str, Image.Image, np.ndarray, Path]): The input image.
                Can be a file path, Path object, PIL Image, or NumPy array.

        Returns:
            Tuple[int, int]: Image dimensions as (width, height).

        Raises:
            FileNotFoundError: If `image_input` is a path and the file is not found.
            UnidentifiedImageError: If PIL cannot open or identify the image file.
            TypeError: If `image_input` type is unsupported for direct size reading.
        """
        img: Image.Image
        if isinstance(image_input, (str, Path)):
            try:
                img = Image.open(str(image_input))
            except FileNotFoundError:
                logger.error(f"Image file not found for size retrieval: {image_input}")
                raise
            except UnidentifiedImageError:
                logger.error(f"Cannot open or identify image file for size retrieval: {image_input}")
                raise
            except Exception as e:
                logger.error(f"Error opening image {image_input} for size retrieval: {e}")
                raise
        elif isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, np.ndarray):
            # For NumPy array, height and width are typically shape[0] and shape[1]
            if image_input.ndim >= 2:
                return image_input.shape[1], image_input.shape[0] # (width, height)
            else:
                raise ValueError("NumPy array has insufficient dimensions to get image size.")
        else:
            raise TypeError(f"Unsupported input type for get_image_size: {type(image_input)}.")
        return img.size # PIL .size returns (width, height)

# Standalone utility functions (potentially using ImageProcessor internally)

DEFAULT_MAX_SIZE = 512 # Default consistent with ImageProcessor

def load_image_as_tensor(image_path: Union[str, Path], max_size: int = DEFAULT_MAX_SIZE, model_type: str = 'adain') -> torch.Tensor:
    """
    Loads an image from a path, preprocesses it, and returns it as a tensor.
    Uses ImageProcessor internally.

    Args:
        image_path: Path to the image file.
        max_size: Maximum dimension (width or height) for resizing.
        model_type: Model type ('adain' or 'johnson') for specific preprocessing.

    Returns:
        A PyTorch tensor representing the image.
    """
    processor = ImageProcessor(max_size=max_size)
    logger.debug(f"Loading image {image_path} as tensor with max_size={max_size}, model_type={model_type}")
    return processor.preprocess_image(image_path, model_type=model_type)

def save_tensor_as_image(tensor: torch.Tensor, output_path: Union[str, Path], model_type: str = 'adain'):
    """
    Converts a PyTorch tensor to a PIL Image and saves it to the specified path.
    Uses ImageProcessor internally for postprocessing.

    Args:
        tensor: The PyTorch tensor to save (output of the model).
        output_path: Path to save the image.
        model_type: Model type ('adain' or 'johnson') for specific postprocessing.
    """
    processor = ImageProcessor() # max_size doesn't apply to postprocessing
    pil_image = processor.postprocess_image(tensor, model_type=model_type)

    # Ensure output directory exists
    output_p = Path(output_path)
    if output_p.parent:
        output_p.parent.mkdir(parents=True, exist_ok=True)

    pil_image.save(output_path)
    logger.debug(f"Saved tensor as image to {output_path} (model_type: {model_type})")

def get_image_size(image_path: Union[str, Path]) -> Tuple[int, int]:
    """
    Retrieves the dimensions (width, height) of an image from a path.
    Uses ImageProcessor internally.

    Args:
        image_path: Path to the image file.

    Returns:
        A tuple (width, height).
    """
    processor = ImageProcessor() # max_size not strictly needed for just getting size
    return processor.get_image_size(image_path) # Delegates to the class method

# It seems coral_preserve_color was specific to an older direct AdaIN implementation.
# The StyleTransferModel.transfer_style method now handles color preservation internally
# when its `preserve_color` flag is True.
# If a standalone coral_preserve_color is absolutely needed for other purposes,
# its logic would need to be defined here. For now, we assume it's covered by StyleTransferModel.

# Example usage (optional, for testing this module directly)
if __name__ == '__main__':
    # Configure logger for standalone testing
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Testing image_processing.py standalone functions...")

    # Create a dummy image for testing (requires Pillow)
    dummy_image_path = "_temp_dummy_test_image.png"
    try:
        img = Image.new('RGB', (600, 400), color = 'red')
        img.save(dummy_image_path)

        # Test get_image_size
        width, height = get_image_size(dummy_image_path)
        logger.info(f"get_image_size({dummy_image_path}): width={width}, height={height}")
        assert width == 600 and height == 400

        # Test load_image_as_tensor (AdaIN type)
        tensor_adain = load_image_as_tensor(dummy_image_path, max_size=300, model_type='adain')
        logger.info(f"load_image_as_tensor (adain, max_size=300) result shape: {tensor_adain.shape}") # Expected (1, 3, H, W) e.g. (1,3,200,300)
        assert tensor_adain.ndim == 4 and tensor_adain.shape[0] == 1 and tensor_adain.shape[1] == 3

        # Test load_image_as_tensor (Johnson type)
        tensor_johnson = load_image_as_tensor(dummy_image_path, max_size=256, model_type='johnson')
        logger.info(f"load_image_as_tensor (johnson, max_size=256) result shape: {tensor_johnson.shape}")
        assert tensor_johnson.ndim == 4 and tensor_johnson.shape[0] == 1 and tensor_johnson.shape[1] == 3


        # Test save_tensor_as_image
        output_image_path_adain = "_temp_dummy_output_adain.png"
        save_tensor_as_image(tensor_adain, output_image_path_adain, model_type='adain')
        logger.info(f"Saved AdaIN tensor to {output_image_path_adain}")
        assert Path(output_image_path_adain).exists()

        output_image_path_johnson = "_temp_dummy_output_johnson.png"
        save_tensor_as_image(tensor_johnson, output_image_path_johnson, model_type='johnson')
        logger.info(f"Saved Johnson tensor to {output_image_path_johnson}")
        assert Path(output_image_path_johnson).exists()

        logger.info("Standalone tests completed successfully.")

    except ImportError:
        logger.warning("Pillow not fully available. Skipping some standalone tests that create/save images.")
    except Exception as e:
        logger.error(f"Error during standalone testing: {e}", exc_info=True)
    finally:
        # Clean up dummy files
        for p in [dummy_image_path, output_image_path_adain, output_image_path_johnson]:
            if Path(p).exists():
                Path(p).unlink()
                logger.debug(f"Cleaned up {p}")