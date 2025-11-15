import torch
import numpy as np
from escnn import nn, gspaces



def make_group_elements(r2_act, thetas):
    """
    Build a list of group elements for escnn gspaces:
      - SO(2):     element(theta)
      - O(2):      element(flip, theta) with flip in {0,1}
    thetas are radians.
    """
    G = r2_act.fibergroup
    elems = []
    # Try SO(2)-style first (single parameter: angle)
    try:
        elems = [G.element(float(t)) for t in thetas]
        return elems
    except TypeError:
        pass

    for flip in (0, 1):
        elems += [G.element((flip, float(t))) for t in thetas]
    return elems


@torch.inference_mode()
def rel_err(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    diff  = torch.linalg.vector_norm(x - y, dim=1)
    denom = torch.maximum(torch.linalg.vector_norm(x, dim=1),
                          torch.linalg.vector_norm(y, dim=1))
    denom = torch.clamp(denom, min=1e-12)
    return diff / denom

@torch.inference_mode()
def chech_invariance_batch_r2(
    x: torch.Tensor,
    model,
    num_samples: int = 16,
    chunk_size: int = 4,
):
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
    y_ref = model.forward_features(x)
    y_ref_tensor = y_ref.tensor if isinstance(y_ref, nn.GeometricTensor) else y_ref

    # Build transformed inputs (GeoTensor -> transform)
    x_geo = nn.GeometricTensor(x, model.input_type)

    errs = []

    B = x.shape[0]
    chunk_size = max(1, int(chunk_size))

    for start in range(0, len(elems), chunk_size):
        chunk_elems = elems[start:start + chunk_size]
        x_rot_list = [x_geo.transform(g).tensor for g in chunk_elems]
        x_rot_batch = torch.cat(x_rot_list, dim=0)

        y_rot_batch = model.forward_features(
            nn.GeometricTensor(x_rot_batch, model.input_type)
        )
        y_rot_batch_tensor = (
            y_rot_batch.tensor if isinstance(y_rot_batch, nn.GeometricTensor) else y_rot_batch
        )

        for y_rotated in torch.split(y_rot_batch_tensor, B, dim=0):
            e = rel_err(y_rotated, y_ref_tensor).mean()
            errs.append(e)

    errs = torch.stack(errs).detach().cpu().numpy() 
    return thetas, errs



@torch.inference_mode()
def check_equivariance_batch_r2(
    x: torch.Tensor,
    model,
    num_samples: int = 16,
    chunk_size: int = 4,
    layer=None
):
    """
    Vectorized equivariance test on the *equivariant* feature map returned by model.forward_features.
    Returns: thetas (np.ndarray), errors_per_theta (np.ndarray)
    """
    device = next(model.parameters()).device
    x = x.to(device, non_blocking=True)

    r2_act = getattr(model, "r2_act")
    if layer is not None:
        model.init_nth_layer(layer)
        x = model.forward_upto_nth_layer(x)
        input_tupe = getattr(model, "eq_layers")[layer - 1].in_type
        out_type = getattr(model, "eq_layers")[layer - 1].out_type
        inference = model.forward_nth_layer

    else:
        input_tupe = model.input_type
        out_type = model.output_type
        inference = model.forward_features
    thetas = np.linspace(0.0, 2*np.pi, num_samples, endpoint=False)
    elems = make_group_elements(r2_act, thetas)

    # Reference features
    y_ref = inference(x)
    if isinstance(y_ref, nn.GeometricTensor):
        y_ref_geo = y_ref
    else:
        if out_type is None:
            raise ValueError("Model must expose `output_type` when returning plain tensors.")
        y_ref_geo = nn.GeometricTensor(y_ref, out_type)

    # Build transformed inputs (GeoTensor -> transform)
    if isinstance(x, nn.GeometricTensor):
        x_geo = x
    else:
        x_geo = nn.GeometricTensor(x, input_tupe)

    errs = []

    B = x.shape[0]
    chunk_size = max(1, int(chunk_size))

    for start in range(0, len(elems), chunk_size):
        chunk_elems = elems[start:start + chunk_size]
        x_rot_list = [x_geo.transform(g).tensor for g in chunk_elems]
        x_rot_batch = torch.cat(x_rot_list, dim=0)

        y_rot_batch = inference(
            nn.GeometricTensor(x_rot_batch, input_tupe)
        )
        y_rot_batch_tensor = (
            y_rot_batch.tensor if isinstance(y_rot_batch, nn.GeometricTensor) else y_rot_batch
        )

        y_ref_rot_list = [y_ref_geo.transform(g).tensor for g in chunk_elems]
        y_ref_rot_batch = torch.cat(y_ref_rot_list, dim=0)
        for y_rotated, y_ref_rotated in zip(
            torch.split(y_rot_batch_tensor, B, dim=0),
            torch.split(y_ref_rot_batch, B, dim=0)
        ):
            e = rel_err(y_rotated, y_ref_rotated).mean()
            errs.append(e)


    errs = torch.stack(errs).detach().cpu().numpy() 
    return thetas, errs
