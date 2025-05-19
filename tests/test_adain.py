import pytest
import torch
import numpy as np
from style_transfer.models.adain import AdaIN, StyleTransferModel

def test_adain_layer(device):
    """Test the AdaIN layer functionality."""
    adain = AdaIN().to(device)
    content = torch.randn(1, 512, 32, 32).to(device)
    style = torch.randn(1, 512, 32, 32).to(device)

    # Test forward pass
    output = adain(content, style)

    assert output.shape == content.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

    # Test with different batch sizes
    content = torch.randn(4, 512, 32, 32).to(device)
    style = torch.randn(1, 512, 32, 32).to(device)
    output = adain(content, style)
    assert output.shape == content.shape

def test_encoder(device):
    """Test the encoder network."""
    model = StyleTransferModel().to(device)
    x = torch.randn(1, 3, 256, 256).to(device)

    # Test feature extraction
    features = model.encode(x)
    assert features.shape[1] == 512  # VGG19 features at relu4_1

    # Test with different input sizes
    x = torch.randn(1, 3, 512, 512).to(device)
    features = model.encode(x)
    assert features.shape[1] == 512

def test_decoder(device):
    """Test the decoder network."""
    model = StyleTransferModel().to(device)
    x = torch.randn(1, 512, 32, 32).to(device)

    # Test decoding
    output = model.decode(x)
    assert output.shape[1] == 3  # RGB output
    assert output.shape[2] == 256  # Upsampled size
    assert output.shape[3] == 256

def test_style_transfer(model, device):
    """Test the complete style transfer process."""
    content = torch.randn(1, 3, 256, 256).to(device)
    style = torch.randn(1, 3, 256, 256).to(device)

    # Test with different alpha values
    for alpha in [0.0, 0.5, 1.0]:
        output = model.transfer_style(content, style, alpha=alpha)
        assert output.shape == content.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    # Test color preservation
    output = model.transfer_style(content, style, preserve_color=True)
    assert output.shape == content.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_model_consistency(model, device):
    """Test model consistency across different runs."""
    content = torch.randn(1, 3, 256, 256).to(device)
    style = torch.randn(1, 3, 256, 256).to(device)

    # Test deterministic output
    torch.manual_seed(42)
    output1 = model.transfer_style(content, style)

    torch.manual_seed(42)
    output2 = model.transfer_style(content, style)

    assert torch.allclose(output1, output2)

def test_model_gradients(model, device):
    """Test that encoder parameters are frozen."""
    for name, param in model.encoder.named_parameters():
        assert not param.requires_grad, f"Parameter {name} should be frozen"

    # Test that decoder parameters are trainable
    for name, param in model.decoder.named_parameters():
        assert param.requires_grad, f"Parameter {name} should be trainable"