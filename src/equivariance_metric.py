import torch
import numpy as np
from escnn import nn as escnn_nn

@torch.inference_mode()
def rel_err(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.flatten(1); y = y.flatten(1)
    diff  = torch.linalg.vector_norm(x - y, dim=1)
    denom = torch.maximum(torch.linalg.vector_norm(x, dim=1),
                          torch.linalg.vector_norm(y, dim=1))
    denom = torch.clamp(denom, min=1e-12)
    return diff / denom

@torch.inference_mode()
def check_equivariance_batch(x: torch.Tensor, model, num_samples: int = 16, chunk: int = 0):
    """
    Vectorized equivariance test on the *equivariant* feature map returned by model.forward_features.
    Returns: thetas (np.ndarray), errors_per_theta (np.ndarray)
    """
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device, non_blocking=True)

    r2_act = getattr(model, "r2_act")
    thetas = np.linspace(0.0, 2*np.pi, num_samples, endpoint=False)
    elems  = [r2_act.fibergroup.element(float(t)) for t in thetas]

    # Reference features
    y_ref = model.forward_features(x)  # GeometricTensor

    # Build transformed inputs (GeoTensor -> transform)
    x_geo = escnn_nn.GeometricTensor(x, model.input_type)
    x_list = [x_geo.transform(g).tensor for g in elems]
    xb = escnn_nn.GeometricTensor(torch.cat(x_list, dim=0), model.input_type)

    # Forward on rotated inputs (optionally chunked)
    if chunk and chunk > 0:
        outs = []
        for i in range(0, xb.tensor.size(0), chunk):
            slice_geo = escnn_nn.GeometricTensor(xb.tensor[i:i+chunk], xb.type)
            outs.append(model.forward_features(slice_geo).tensor)
        y_rot_tensor = torch.cat(outs, dim=0)
    else:
        y_rot_tensor = model.forward_features(xb).tensor

    # Transform reference features
    y_ref_list = [y_ref.transform(g).tensor for g in elems]
    y_ref_b = torch.cat(y_ref_list, dim=0)

    # Relative error per angle (averaged over batch)
    B = x.shape[0]
    errs = rel_err(y_rot_tensor, y_ref_b).view(num_samples, B).mean(dim=1)
    return thetas, errs.detach().cpu().numpy()

@torch.inference_mode()
def logits_invariance_error(model, x, angles=(0, 90, 180, 270)):
    """
    Relative invariance error on logits after the invariant head.
    """
    from torchvision.transforms.functional import rotate, InterpolationMode
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device, non_blocking=True)

    base = model(x)  # (B, C)
    errs = {}
    for a in angles:
        xr = rotate(x, a, interpolation=InterpolationMode.BILINEAR)
        yr = model(xr)
        errs[a] = rel_err(base, yr).mean().item()
    return errs
