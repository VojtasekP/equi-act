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
    y_ref = model.forward_invar_features(x)
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

        y_rot_batch = model.forward_invar_features(
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
def check_invariance_resnet(
    x: torch.Tensor,
    model,
    num_samples: int = 16,
    chunk_size: int = 4,
):
    """Measure invariance of ResNet features to rotation using torchvision transforms.

    Returns: thetas (np.ndarray), errors_per_theta (np.ndarray)
    """
    import torchvision.transforms.functional as TF

    device = next(model.parameters()).device
    x = x.to(device, non_blocking=True)

    # Generate rotation angles (radians)
    thetas = np.linspace(0.0, 2*np.pi, num_samples, endpoint=False)
    degrees = np.degrees(thetas)

    # Extract reference features
    y_ref = model.forward_invar_features(x)  # (B, 512)

    errs = []
    B = x.shape[0]
    chunk_size = max(1, int(chunk_size))

    # Process rotations in chunks
    for start in range(0, len(degrees), chunk_size):
        chunk_degrees = degrees[start:start + chunk_size]

        # Rotate inputs using torchvision
        x_rot_list = [
            TF.rotate(
                x,
                angle=float(angle),
                interpolation=TF.InterpolationMode.BILINEAR,
                expand=False,
                fill=0
            )
            for angle in chunk_degrees
        ]
        x_rot_batch = torch.cat(x_rot_list, dim=0)

        # Extract features for rotated inputs
        y_rot_batch = model.forward_invar_features(x_rot_batch)

        # Compute relative errors
        for y_rotated in torch.split(y_rot_batch, B, dim=0):
            e = rel_err(y_rotated, y_ref).mean()
            errs.append(e)

    errs = torch.stack(errs).detach().cpu().numpy()
    return thetas, errs



@torch.inference_mode()
def check_activation_equivariance_r2(
    x: torch.Tensor,
    model,
    num_samples: int = 16,
    chunk_size: int = 4,
    layer: int = None
):
    """
    Test equivariance of ONLY the activation function in a specific layer.

    This function:
    1. Forwards through conv + batch norm (but NOT activation) for both original and rotated inputs
    2. Applies activation separately
    3. Tests if: activation(Tx) ≈ T(activation(x))

    Args:
        x: Input batch [B, C, H, W]
        model: The equivariant network
        num_samples: Number of rotation angles to test
        chunk_size: Process rotations in chunks for memory efficiency
        layer: 1-based layer index (required)

    Returns:
        thetas: np.ndarray of rotation angles in radians
        errs: np.ndarray of equivariance errors per angle
    """
    if layer is None:
        raise ValueError("layer parameter is required for activation equivariance test")

    device = next(model.parameters()).device
    x = x.to(device, non_blocking=True)

    r2_act = getattr(model, "r2_act")

    # Get the layer structure
    eq_layers = getattr(model, "eq_layers")
    target_layer = eq_layers[layer - 1]

    # Check if this is a SequentialModule (block with multiple components)
    if not hasattr(target_layer, '_modules'):
        raise ValueError(f"Layer {layer} does not have a sequential structure with separable activation")

    # Split layer into pre-activation and activation parts
    # Typical structure: [conv, bn, activation]
    layer_modules = list(target_layer.children())

    # Find the activation module (last module that is an activation type)
    activation_idx = None
    activation_module = None

    from escnn.nn import (FourierPointwise, NormNonLinearity, GatedNonLinearity1,
                          GatedNonLinearity2, MultipleModule, ReLU, ELU)
    activation_types = (FourierPointwise, NormNonLinearity, GatedNonLinearity1,
                        GatedNonLinearity2, MultipleModule, ReLU, ELU)

    for idx in range(len(layer_modules) - 1, -1, -1):
        if isinstance(layer_modules[idx], activation_types):
            activation_idx = idx
            activation_module = layer_modules[idx]
            break

    if activation_module is None:
        raise ValueError(f"No activation module found in layer {layer}")

    # Create a partial forward function (up to but not including activation)
    pre_activation_modules = layer_modules[:activation_idx]

    def forward_pre_activation(input_geo):
        """Forward through conv + bn only"""
        x_temp = input_geo
        for module in pre_activation_modules:
            x_temp = module(x_temp)
        return x_temp

    # Forward to the start of this layer
    model.init_nth_layer(layer)
    x = model.forward_upto_nth_layer(x)
    input_type = target_layer.in_type

    if not isinstance(x, nn.GeometricTensor):
        x = nn.GeometricTensor(x, input_type)

    # Get features before activation
    feat_before_act = forward_pre_activation(x)
    intermediate_type = feat_before_act.type

    thetas = np.linspace(0.0, 2*np.pi, num_samples, endpoint=False)
    elems = make_group_elements(r2_act, thetas)

    # Apply activation to reference features
    y_ref_after_act = activation_module(feat_before_act)

    errs = []
    B = x.tensor.shape[0]
    chunk_size = max(1, int(chunk_size))

    for start in range(0, len(elems), chunk_size):
        chunk_elems = elems[start:start + chunk_size]

        # Path 1: Rotate input, then forward through pre-activation, then activate
        x_rot_list = [x.transform(g).tensor for g in chunk_elems]
        x_rot_batch = torch.cat(x_rot_list, dim=0)
        x_rot_geo = nn.GeometricTensor(x_rot_batch, input_type)

        feat_rot_before_act = forward_pre_activation(x_rot_geo)
        y_rot_after_act = activation_module(feat_rot_before_act)

        # Path 2: Apply activation to reference, then rotate the result
        y_ref_rot_list = [y_ref_after_act.transform(g).tensor for g in chunk_elems]
        y_ref_rot_batch = torch.cat(y_ref_rot_list, dim=0)

        # Compare the two paths
        y_rot_tensor = y_rot_after_act.tensor if isinstance(y_rot_after_act, nn.GeometricTensor) else y_rot_after_act

        for y_rotated, y_ref_rotated in zip(
            torch.split(y_rot_tensor, B, dim=0),
            torch.split(y_ref_rot_batch, B, dim=0)
        ):
            e = rel_err(y_rotated, y_ref_rotated).mean()
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
        out_type = getattr(model, "output_type", None)
        if out_type is None:
            eq_layers = getattr(model, "eq_layers", None)
            if eq_layers:
                out_type = eq_layers[-1].out_type
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
