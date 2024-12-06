import torch
import pytest
from model.model import Net

def test_model_structure():
    model = Net()
    # Test model exists
    assert model is not None

def test_model_forward():
    model = Net()
    # Create random input tensor
    x = torch.randn(1, 1, 28, 28)
    # Forward pass
    output = model(x)
    # Check output shape
    assert output.shape == (1, 10)

def test_model_parameters():
    model = Net()
    # Count number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    # Check if number of parameters matches expected
    assert total_params == 13272

def test_output_range():
    model = Net()
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    # Test if output is a valid probability distribution (log_softmax)
    assert torch.allclose(torch.exp(output).sum(dim=1), torch.tensor([1.0]))

def test_batch_processing():
    model = Net()
    # Test with a batch of 32 images
    batch_size = 32
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)
    assert output.shape == (batch_size, 10) 