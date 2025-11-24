import numpy as np
from escnn.nn import FieldType, R2Conv, R3Conv
from escnn.gspaces import rot2dOnR2, flipRot2dOnR2
def adjust_channels(
    C: int,
    group,                         # r1.gspace (an escnn gspace, e.g. rot2dOnR2(...))
    kernel_size: int,
    activation_type:str,
    max_frequency=3,
    layer_index: int = 1,
    last_layer: bool = False,
    mnist: bool = True,
    volumetric:bool = False) -> int:
    """
    Compute the channel count to use given input channels C, the group (gspace),
    and the architectural knobs, mirroring the scaling in your snippet.

    Args:
        C: nominal channels you'd like to use before scaling
        gc: escnn gspace (e.g. model.r2_act)
        kernel_size: convolution kernel size (s in your code)
        padding: convolution padding
        fix_param: if True and not invariant_map and layer_index>1, scale C to keep params ~constant
        invariant_map: if True, shrink C to preserve same number of OUTPUT channels after invariant map
        layer_index: 1-based layer index
        frequencies_cutoff, sigma, maximum_offset: forwarded to the temporary R2Conv when needed

    Returns:
        int: adjusted channel count
    """
    C = int(C)
    print(f"Adjusting channels layer {layer_index}, C={C}")
    # ---- build the "irreps list" as in your code, excluding the trivial ----
    irreps = []
    for irr in group.irreps:
        if irr.is_trivial():
            continue
        mult = int(irr.size // irr.sum_of_squares_constituents)
        irreps.extend([irr] * mult)
    ref_group = rot2dOnR2(maximum_frequency=max_frequency)
    irreps_ref = []
    for irr in ref_group.irreps:
        if irr.is_trivial():
            continue
        mult = int(irr.size // irr.sum_of_squares_constituents)
        irreps_ref.extend([irr] * mult)
    # ---- invariant map case: keep number of *output* channels constant across groups ----
    if last_layer:
        size = 0
        size = sum((irrep.size // irrep.sum_of_squares_constituents) for irrep in group.irreps)
        print(f"New channel size {int(round(C/size))}")
        return max(1, int(round(C / size)))

    # ---- parameter-fix case: keep params roughly constant by compensating kernel basis dimension ----
    if layer_index > 1:
        # r_in and r_out are [trivial] + irreps
        r_in_ref  = FieldType(ref_group, [ref_group.trivial_repr] + list(irreps_ref))
        r_out_ref = FieldType(ref_group, [ref_group.trivial_repr] + list(irreps_ref))
        if activation_type in ['fourier_relu', 'fourier_elu', 'norm_relu', 'normbn_relu','norm_squash', 'non_equi_relu']:
            r_in = FieldType(group, [group.trivial_repr]  + irreps)
            r_out = FieldType(group, [group.trivial_repr] + irreps)
        elif activation_type in ['gated_shared_sigmoid']:
            r_in = FieldType(group, [group.trivial_repr] * 2 + irreps)
            r_out = FieldType(group, [group.trivial_repr] + irreps)
        elif activation_type in ['gated_sigmoid', 'non_equi_relu']:
            I = len(irreps)

            r_in = FieldType(group, [group.trivial_repr] * (I + 1) + irreps)
            r_out = FieldType(group, [group.trivial_repr] + irreps)
        if volumetric:
            tmp = R3Conv(
                r_in, r_out, kernel_size,sigma=0.6
            )
        else:
            tmp = R2Conv(
                r_in, r_out, kernel_size,sigma=0.6
            )
        # basis dimension of constrained kernel space
        t = float(tmp.basisexpansion.dimension())
        print(f"t: {t}")

        # normalize w.r.t. the reference SFCNN factor you had:
        # 16 * s^2 * 3 / 4  (exactly as in your snippet)
        if mnist:
            denom = 16.0 * (kernel_size ** 2) * 3.0 / 4.0 # this is from E2 paper, wrt to the 16GCNN
        else:
            if volumetric:
                tmp_2 = R3Conv(r_in_ref, r_out_ref, kernel_size,sigma=0.6) # this is the reference conv to keep params constant wrt to the fourier/norm activations
            else:
                tmp_2 = R2Conv(r_in_ref, r_out_ref, kernel_size,sigma=0.6) # this is the reference conv to keep params constant wrt to the fourier/norm activations
            denom = float(tmp_2.basisexpansion.dimension()) # num of params for the reference conv
        t /= denom if denom > 0 else 1.0 # relative number of params

        scale = 1.0 / max(np.sqrt(t), 1e-12)
        print(f"New channel size {int(round(C * scale))}")
        return max(1, int(round(C * scale)))


    print(f"New channel size {int(round(C))}")
    return max(1, int(C))


if __name__ == "__main__":
    # Example usage

    max_frequency = 3
    gspace = rot2dOnR2(maximum_frequency=3)  # example group

    # gspace = flipRot2dOnR2(maximum_frequency=max_frequency)  # example group
    nominal_channels = 24
    channels = [16, 24, 32, 32, 48, 64]
    kernels =  [7, 5, 5, 5, 5, 5]

    for layer_idx, (C, k) in enumerate(zip(channels, kernels), start=1):
        adjusted_C = adjust_channels(
            C,
            gspace,
            kernel_size=k,
            activation_type='gated_sigmoid',
            max_frequency=max_frequency,
            layer_index=layer_idx,
            last_layer=(layer_idx == len(channels))
        )
        print(f"Layer {layer_idx}: nominal channels={C}, adjusted channels={adjusted_C}")