import pytest
import torch
from style_transfer.utils.device_utils import get_device, clear_gpu_memory, get_optimal_batch_size
import logging

def test_get_device():
    """Test device selection functionality."""
    # Test automatic device selection
    device = get_device()
    assert isinstance(device, torch.device)

    # Test explicit CPU selection
    device = get_device('cpu')
    assert device.type == 'cpu'

    # Test explicit CUDA selection if available
    if torch.cuda.is_available():
        device = get_device('cuda')
        assert device.type == 'cuda'
    else:
        with pytest.raises(RuntimeError):
            get_device('cuda')

def test_clear_gpu_memory():
    """Test GPU memory clearing functionality more robustly."""
    if torch.cuda.is_available():
        device = torch.device("cuda")

        # Establish a baseline memory usage at the start of this specific test.
        # Call clear_gpu_memory() first to try and get a cleaner baseline.
        clear_gpu_memory()
        memory_at_start_of_test = torch.cuda.memory_allocated(device)

        # Allocate a moderately sized tensor (1000*1000 float32 = 4,000,000 bytes)
        x = torch.randn(1000, 1000, device=device)
        actual_tensor_bytes = x.nelement() * x.element_size()
        memory_after_alloc = torch.cuda.memory_allocated(device)

        assert memory_after_alloc > memory_at_start_of_test, \
            f"Memory did not increase after allocation. Start: {memory_at_start_of_test}, After Alloc: {memory_after_alloc}"

        assert memory_after_alloc >= memory_at_start_of_test + actual_tensor_bytes, \
             f"Memory increase ({memory_after_alloc - memory_at_start_of_test} bytes) is less than actual tensor data size ({actual_tensor_bytes} bytes)."

        # Delete the tensor and then call the function under test
        del x
        clear_gpu_memory()

        final_memory = torch.cuda.memory_allocated(device)

        # After deleting the tensor and clearing cache, memory should ideally return to near its state
        # before this specific tensor was allocated. Allow a small fixed overhead for PyTorch internal state
        # or fragmentation that might not be perfectly cleaned.
        allowed_overhead = 1 * 1024 * 1024  # 1MB overhead
        assert final_memory <= memory_at_start_of_test + allowed_overhead, \
            f"GPU memory not cleared effectively. \
            Start of Test: {memory_at_start_of_test}, \
            After Alloc: {memory_after_alloc}, \
            Final: {final_memory}, \
            Allowed Final (Start + Overhead): {memory_at_start_of_test + allowed_overhead}"

def test_get_optimal_batch_size():
    """Test optimal batch size calculation."""
    device = get_device()

    # Test CPU case
    if device.type == 'cpu':
        batch_size = get_optimal_batch_size(device, model_size=100)
        assert batch_size == 1

    # Test GPU case
    if device.type == 'cuda':
        # Test with different model sizes
        for model_size in [100, 500, 1000]:
            batch_size = get_optimal_batch_size(device, model_size)
            assert 1 <= batch_size <= 16  # Adjusted cap to 16
            assert isinstance(batch_size, int)

def test_device_consistency():
    """Test that device selection is consistent."""
    device1 = get_device()
    device2 = get_device()
    assert device1.type == device2.type

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_memory_management(device, model):
    """Test that GPU memory is managed effectively, including model memory."""
    if device.type == 'cuda':
        # Model is already on device by the 'model' fixture which calls model.to(device)
        initial_memory_with_model = torch.cuda.memory_allocated(device)

        # Allocate an additional large tensor
        temp_tensor_size = (1024, 1024, 4)
        temp_tensor = torch.randn(temp_tensor_size, dtype=torch.float32, device=device)
        actual_temp_tensor_bytes = temp_tensor.nelement() * temp_tensor.element_size()
        memory_after_temp_tensor = torch.cuda.memory_allocated(device)

        # Ensure memory increased after allocating the temporary tensor
        assert memory_after_temp_tensor > initial_memory_with_model, \
               "Memory did not increase after allocating a temporary tensor."
        assert memory_after_temp_tensor >= initial_memory_with_model + actual_temp_tensor_bytes, \
               f"Memory increase ({memory_after_temp_tensor - initial_memory_with_model}) is less than the temporary tensor size ({actual_temp_tensor_bytes})."

        # Delete the temporary tensor and clear GPU memory
        del temp_tensor
        clear_gpu_memory() # Explicitly call the function we are testing

        final_memory = torch.cuda.memory_allocated(device)

        # Assert that memory has returned to a level close to before allocating the temp tensor
        # Allow a small fixed overhead (e.g., 1MB) for PyTorch internals/fragmentation
        allowed_overhead = 1 * 1024 * 1024
        assert final_memory <= initial_memory_with_model + allowed_overhead, \
            f"GPU memory not cleared effectively. \
            Initial (with model): {initial_memory_with_model}, \
            After temp tensor: {memory_after_temp_tensor}, \
            Final: {final_memory}, \
            Allowed final: {initial_memory_with_model + allowed_overhead}"

def test_batch_size_limits():
    """Test that batch size calculation respects limits."""
    device = get_device()

    if device.type == 'cuda':
        # Test with very large model size
        batch_size = get_optimal_batch_size(device, model_size=10000)
        assert batch_size == 1

        # Test with very small model size
        batch_size = get_optimal_batch_size(device, model_size=1)
        assert 1 <= batch_size <= 16  # Adjusted cap to 16

def test_get_device_cuda_unavailable(mocker):
    """Test get_device when CUDA is unavailable."""
    mocker.patch('torch.cuda.is_available', return_value=False)
    # Patching get_device_properties as it would be called if is_available was true,
    # though it won't be in this mocked scenario. Adding for completeness if logic changes.
    mocker.patch('torch.cuda.get_device_properties')
    logger_info_spy = mocker.spy(logging.getLogger('style_transfer.utils.device_utils'), 'info')

    device = get_device()
    assert device.type == 'cpu'
    logger_info_spy.assert_any_call("CUDA not available, using CPU")

def test_get_optimal_batch_size_cpu():
    """Test get_optimal_batch_size specifically for CPU device."""
    cpu_device = torch.device('cpu')
    batch_size = get_optimal_batch_size(cpu_device, model_size=100)
    assert batch_size == 1, "Batch size for CPU should always be 1 as per current implementation."

def test_get_optimal_batch_size_gpu_large_model(device):
    """Test get_optimal_batch_size for GPU with a very large model."""
    if device.type == 'cuda':
        batch_size = get_optimal_batch_size(device, model_size=2000) # > 1000MB
        assert batch_size == 1, "Batch size for very large model on GPU should be 1."
    else:
        pytest.skip("Skipping GPU specific test for large model as CUDA is not the active device type.")