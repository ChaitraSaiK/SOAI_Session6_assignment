import torch
import pytest
from Session6_assignment import Net

def test_model_output_shape():
    model = Net()
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 28, 28)
    output = model(input_tensor)
    assert output.shape == (batch_size, 10), "Output shape is incorrect"

def test_parameter_count():
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, "Model has too many parameters"

def test_forward_pass():
    model = Net()
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 28, 28)
    try:
        output = model(input_tensor)
        assert not torch.isnan(output).any(), "Model output contains NaN values"
    except Exception as e:
        pytest.fail(f"Forward pass failed with error: {str(e)}") 