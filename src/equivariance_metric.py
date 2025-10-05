import numpy as np
from escnn import nn as escnn_nn   # to avoid shadowing torch.nn as nn
import torch


@torch.inference_mode()
def rel_err(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.flatten(1); y = y.flatten(1)
    diff  = torch.linalg.vector_norm(x - y, dim=1)
    denom = torch.maximum(torch.linalg.vector_norm(x, dim=1),
                          torch.linalg.vector_norm(y, dim=1))
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    return diff / denom


@torch.inference_mode()
def check_equivariance_batch(x, model, r2_act, num_samples=32):
    """
    Vectorized equivariance test on the *equivariant* feature map returned by forward_features.
    x: (B,C,H,W) torch.Tensor
    r2_act: gspaces.rot2dOnR2(-1, maximum_frequency=...)
    Returns (thetas, mean_errors_per_theta)
    """
    # sample distinct group elements (avoid duplicate 0 ≡ 2π)
    thetas = np.linspace(0.0, 2*np.pi, num_samples, endpoint=False)
    elems  = [r2_act.fibergroup.element(float(t)) for t in thetas]

    # forward once to get reference feature GeometricTensor
    y_ref = model.forward_features(x)            # GeometricTensor
    # build transformed inputs
    x_geo = escnn_nn.GeometricTensor(x, model.input_type)
    xs = [x_geo.transform(g) for g in elems]
    xb = escnn_nn.GeometricTensor(torch.cat([t.tensor for t in xs], dim=0), model.input_type)

    # single batched pass on rotated inputs
    y_rot = model.forward_features(xb)           # GeometricTensor

    # transform reference output and stack
    ys = [y_ref.transform(g) for g in elems]
    y_ref_b = escnn_nn.GeometricTensor(torch.cat([t.tensor for t in ys], dim=0), y_ref.type)

    # relative error per sample
    errs = rel_err(y_rot.tensor, y_ref_b.tensor)  # shape (num_samples*B,)
    B = x.shape[0]
    return thetas, errs.view(num_samples, B).mean(dim=1).cpu().numpy()
import torch
import numpy as np
from torchvision.transforms.functional import rotate, InterpolationMode



@torch.inference_mode()
def logits_invariance_error(model, x, angles=(0, 90, 180, 270)):
    """
    Relative invariance error on logits. x: (B,C,H,W) torch.Tensor on correct device.
    Returns dict: {angle_deg: scalar_error}.
    """
    model.eval()
    base = model(x)  # logits (B, num_classes)
    errs = {}
    for a in angles:
        xr = rotate(x, a, interpolation=InterpolationMode.BILINEAR)
        yr = model(xr)
        errs[a] = rel_err(base, yr).mean().item()
    return errs
