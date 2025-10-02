import math
import torch
import pytest
from escnn import nn as escnn_nn
from HNet import HNet

def test_hnet_forward_shapes():
    model = HNet(n_classes=10, max_rot_order=2, channels_per_block=(2, 2), layers_per_block=1,
                 gated=True, kernel_size=3, pool_stride=2, pool_sigma=1.0, invariant_channels=8,
                 use_bn=False, non_linearity='n_relu', grey_scale=True)
    x = torch.randn(4, 1, 29, 29)
    out = model(x)
    assert out.shape == (4, 10)

def test_hnet_rotation_invariance_logits():
    model = HNet(n_classes=7, max_rot_order=2, channels_per_block=(2,), layers_per_block=1,
                 gated=False, kernel_size=3, pool_stride=2, pool_sigma=1.0, invariant_channels=4,
                 use_bn=False, non_linearity='n_relu', grey_scale=True)

    # Build GeometricTensor to rotate the input consistently with the model's input_type. :contentReference[oaicite:12]{index=12}
    r2_act = model.r2_act
    inp_type = model.input_type

    x = torch.randn(2, 1, 29, 29)
    gx = escnn_nn.GeometricTensor(x.clone(), inp_type)

    theta = math.pi * 0.23
    g = r2_act.fibergroup.element(theta)
    x_rot = gx.transform(g).tensor  # raw tensor back for model forward

    y = model(x)
    y_rot = model(x_rot)

    torch.testing.assert_close(y_rot, y, rtol=1e-4, atol=1e-5)
