"""
Utilities for managing PyTorch devices (CPU/GPU) and GPU memory.

This module provides functions to:
- Select the appropriate compute device (CUDA GPU or CPU).
- Clear cached GPU memory.
- Estimate an optimal batch size based on available GPU memory (heuristic).
"""
import torch
import logging
from typing import Optional
import gc # Moved import gc to the top as it's standard practice

logger = logging.getLogger(__name__)

def get_available_devices() -> list[str]:
    """
    Returns a list of available compute devices.
    Checks for CUDA, MPS (for Apple Silicon), and always includes CPU.
    """
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    # Check for MPS (Apple Silicon GPU)
    # hasattr is used for compatibility with older PyTorch versions
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    logger.info(f"Available devices: {devices}")
    return devices

def get_device(device_name: Optional[str] = None) -> torch.device:
    """
    Selects and returns the appropriate PyTorch device for computation.

    Resolution order:
    1. If `device_name` is specified ('cuda' or 'cpu'), that device is used.
    2. If `device_name` is None and a CUDA-enabled GPU is available, 'cuda' is used.
    3. Otherwise, 'cpu' is used.

    Args:
        device_name (Optional[str], optional): Specific device to request ('cuda' or 'cpu').
                                               If None, auto-detects. Defaults to None.

    Returns:
        torch.device: The selected PyTorch device instance.
    """
    if device_name is not None:
        logger.info(f"Device explicitly set to: {device_name}")
        return torch.device(device_name)

    if torch.cuda.is_available():
        # Get the current CUDA device properties
        device_idx = torch.cuda.current_device()
        device_props = torch.cuda.get_device_properties(device_idx)
        logger.info(f"Using GPU: {device_props.name} (CUDA Device {device_idx})")
        logger.info(f"GPU Memory: {device_props.total_memory / 1024**3:.2f} GB")
        return torch.device('cuda')
    else:
        logger.info("CUDA not available or not selected, using CPU.")
        return torch.device('cpu')

def clear_gpu_memory():
    """Attempts to clear cached GPU memory if CUDA is available.

    This function calls `torch.cuda.empty_cache()`, `torch.cuda.synchronize()`,
    and `gc.collect()` multiple times. This is a common practice to try and
    ensure as much memory as possible is freed, although its effectiveness can vary.
    It also resets peak memory statistics.
    """
    if torch.cuda.is_available():
        logger.debug("Clearing GPU memory cache...")
        # Synchronize to ensure all pending operations are complete before clearing cache
        torch.cuda.synchronize()
        # Clear cache
        torch.cuda.empty_cache()
        # Force Python garbage collection
        gc.collect()
        # Synchronize again and clear cache after garbage collection
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # Reset peak memory stats for the current device
        torch.cuda.reset_peak_memory_stats()
        logger.debug("GPU memory cache clearing attempt complete.")
    else:
        logger.debug("CUDA not available, no GPU memory to clear.")

def get_optimal_batch_size(device: torch.device, model_size_mb: int, image_size_mb: int = 4) -> int:
    """
    Estimates an optimal batch size based on available GPU memory and model size.

    This is a heuristic and may require tuning for specific models and image dimensions.
    It attempts to leave some memory headroom.

    Args:
        device (torch.device): The PyTorch device being used.
        model_size_mb (int): Approximate size of the model in Megabytes (MB).
        image_size_mb (int, optional): Estimated memory footprint per image in a batch (MB).
                                     Defaults to 4MB.

    Returns:
        int: An estimated optimal batch size. Returns 1 for CPU or if estimation fails.
    """
    if device.type == 'cuda':
        try:
            # Get total and allocated memory on the current CUDA device
            device_idx = torch.cuda.current_device()
            total_memory_bytes = torch.cuda.get_device_properties(device_idx).total_memory
            allocated_memory_bytes = torch.cuda.memory_allocated(device_idx)
            free_memory_bytes = total_memory_bytes - allocated_memory_bytes

            # Convert model and image sizes to bytes
            model_size_bytes = model_size_mb * 1024**2
            image_size_bytes = image_size_mb * 1024**2

            # Heuristic: leave some headroom (e.g., 20% of free memory or 500MB, whichever is smaller)
            headroom_bytes = min(free_memory_bytes * 0.2, 500 * 1024**2)
            available_for_batches_bytes = free_memory_bytes - model_size_bytes - headroom_bytes

            if available_for_batches_bytes <= 0 or image_size_bytes <= 0:
                logger.warning("Not enough estimated free memory for even one batch after model load and headroom.")
                return 1

            estimated_batch_size = int(available_for_batches_bytes / image_size_bytes)

            # For very large models relative to memory, or very small estimated batch size, be conservative.
            if model_size_mb > 1000:  # If model is larger than 1GB (arbitrary threshold)
                logger.info(f"Large model detected ({model_size_mb}MB), defaulting batch size to 1.")
                return 1

            # Cap batch size at a reasonable maximum (e.g., 16) and ensure at least 1.
            optimal_batch_size = max(1, min(estimated_batch_size, 16))
            logger.info(f"Estimated optimal batch size for GPU: {optimal_batch_size}")
            return optimal_batch_size
        except Exception as e:
            logger.error(f"Error estimating optimal batch size for GPU: {e}. Defaulting to 1.")
            return 1
    else:
        logger.info("Device is CPU, defaulting batch size to 1.")
        return 1  # Default to 1 for CPU