import math
import torch
import pytest
from escnn import nn as escnn_nn
import layers

@pytest.mark.parametrize("builder", ["gated", "norm"])
def test_block_equivariance(r2_act, input_type_gray, irreps, builder):
    # small block so it’s fast
    if builder == "gated":
        block, out_type = layers.make_gated_block(r2_act, input_type_gray, irreps, channels=1, layers_num=1,
                                                  kernel_size=3, pad=1, use_bn=False)
    else:
        block, out_type = layers.make_norm_block(r2_act, input_type_gray, irreps, channels=1, layers_num=1,
                                                 kernel_size=3, pad=1, use_bn=False, non_linearity="n_relu")

    x = torch.randn(1, 1, 29, 29)
    gx = escnn_nn.GeometricTensor(x.clone(), input_type_gray)
    y1 = block(gx)

    # rotate input by some angle and compare with transforming the output
    theta = math.pi * 0.37
    g = r2_act.fibergroup.element(theta)
    y2 = block(gx.transform(g))
    # Now transform y1 itself
    y1g = y1.transform(g)

    torch.testing.assert_close(y2.tensor, y1g.tensor, rtol=1e-4, atol=1e-5)
