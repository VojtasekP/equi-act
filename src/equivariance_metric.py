import torch
import numpy as np
import escnn.nn as nn

@torch.inference_mode()
def rel_err(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    diff  = torch.linalg.vector_norm(x - y, dim=1)
    denom = torch.maximum(torch.linalg.vector_norm(x, dim=1),
                          torch.linalg.vector_norm(y, dim=1))
    denom = torch.clamp(denom, min=1e-12)
    return diff / denom

@torch.inference_mode()
def chech_invariance_batch(x: torch.Tensor, model, num_samples: int = 16):
    """
    Vectorized equivariance test on the *equivariant* feature map returned by model.forward_features.
    Returns: thetas (np.ndarray), errors_per_theta (np.ndarray)
    """
    device = next(model.parameters()).device
    x = x.to(device, non_blocking=True)

    r2_act = getattr(model, "r2_act")
    thetas = np.linspace(0.0, 2*np.pi, num_samples, endpoint=False)
    elems  = [r2_act.fibergroup.element(float(t)) for t in thetas]

    # Reference features
    y_ref = model.forward_features(x)  # GeometricTensor

    # Build transformed inputs (GeoTensor -> transform)
    x_geo = nn.GeometricTensor(x, model.input_type)

    x_rot_list = [x_geo.transform(g).tensor for g in elems]        # list of (B, C, H, W)
    x_rot_batch = torch.cat(x_rot_list, dim=0) 

    y_rot_batch = model.forward_features(
        nn.GeometricTensor(x_rot_batch, model.input_type)
    )

    B = x.shape[0]
    y_rot_list = torch.split(y_rot_batch, B, dim=0)
    errs = []

    for y_rotated in y_rot_list:
        e = rel_err(y_rotated, y_ref).mean()
        errs.append(e)

    errs = torch.stack(errs).detach().cpu().numpy() 
    return thetas, errs

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
