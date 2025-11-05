import torch
import pytest
from escnn import gspaces, nn as escnn_nn

@pytest.fixture(scope="session")
def r2_act():
    # Match your HNet: continuous rotations with maximum_frequency decided later per test
    return gspaces.rot2dOnR2(-1, maximum_frequency=3)

@pytest.fixture(scope="session")
def input_type_gray(r2_act):
    # HNet uses trivial rep for grayscale input when grey_scale=True. :contentReference[oaicite:4]{index=4}
    return escnn_nn.FieldType(r2_act, [r2_act.trivial_repr])

@pytest.fixture(scope="session")
def random_batch():
    torch.manual_seed(0)
    # 8 grayscale 29x29 to match your MnistRot transforms default-ish size (you resize/avgpool anyway). :contentReference[oaicite:5]{index=5}
    return torch.randn(16, 1, 64, 64)

@pytest.fixture(scope="session")
def irreps(r2_act):
    # HNet collects irreps up to max_rot_order. :contentReference[oaicite:6]{index=6}
    max_rot_order = 3
    return [r2_act.irrep(m) for m in range(max_rot_order + 1)]
