import torch
import pytest
from escnn import nn as escnn_nn
import layers  # your file

def test_gated_block_types_and_bn(r2_act, input_type_gray, irreps):
    # Build a small gated block. Should drop gate channels and expose only feature_type. :contentReference[oaicite:7]{index=7}
    block, out_type = layers.make_gated_block(
        act=r2_act,
        in_type=input_type_gray,
        irrep_repr=irreps,
        channels=2,
        layers_num=1,
        kernel_size=3,
        pad=1,
        use_bn=True
    )
    assert isinstance(block, escnn_nn.SequentialModule)
    # escnn modules carry .out_type
    assert hasattr(block, "out_type")
    assert block.out_type == out_type
    # Ensure output has no gate fields left (GatedNonLinearity1 with drop_gates=True). :contentReference[oaicite:8]{index=8}
    # crude check: length of representations equals len(irrep_repr) * channels
    assert len(out_type.representations) == len(irreps) * 2

def test_norm_block_invalid_nonlinearity_raises(r2_act, input_type_gray, irreps):
    with pytest.raises(ValueError):
        layers.make_norm_block(r2_act, input_type_gray, irreps, channels=1, layers_num=1,
                               non_linearity="i_made_this_up")  # should raise. :contentReference[oaicite:9]{index=9}

@pytest.mark.parametrize("nonlin", ["n_relu", "n_sigmoid", "n_softplus", "squash"])
def test_norm_block_forward_shapes(r2_act, input_type_gray, irreps, random_batch, nonlin):
    block, out_type = layers.make_norm_block(
        r2_act, input_type_gray, irreps, channels=1, layers_num=2,
        kernel_size=3, pad=1, use_bn=True, non_linearity=nonlin
    )
    x = escnn_nn.GeometricTensor(random_batch, input_type_gray)
    y = block(x)
    assert isinstance(y, escnn_nn.GeometricTensor)
    assert y.type == out_type
    # spatial dims preserved by padding=1, kernel=3
    assert y.tensor.shape[-2:] == x.tensor.shape[-2:]

def test_fourier_block_uses_quotient_impl(r2_act, input_type_gray, irreps):
    # Your code uses nn.QuotientFourierPointwise. Let’s verify the pipeline runs. :contentReference[oaicite:10]{index=10}
    block, out_type = layers.make_fourier_block(
        r2_act, input_type_gray, irreps, channels=1, layers_num=1, kernel_size=3, pad=1, use_bn=False
    )
    assert isinstance(block, escnn_nn.SequentialModule)
    # sanity: a forward pass
    x = escnn_nn.GeometricTensor(torch.randn(2, 1, 29, 29), input_type_gray)
    y = block(x)
    assert y.type == out_type
