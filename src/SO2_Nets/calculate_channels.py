import numpy as np
from escnn.nn import FieldType, R2Conv
from escnn.gspaces import rot2dOnR2, flipRot2dOnR2
def adjust_channels(
    C: int,
    group,                         # r1.gspace (an escnn gspace, e.g. rot2dOnR2(...))
    kernel_size: int,
    activation_type:str,
    max_frequency=3,
    layer_index: int = 1,
    last_layer: bool = False,

) -> int:
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
    if group == rot2dOnR2(maximum_frequency=max_frequency):
        for m in range(1, max_frequency + 1):
            irr = group.irrep(m)
            mult = int(irr.size // irr.sum_of_squares_constituents)
            irreps.extend([irr] * mult)
    elif group == flipRot2dOnR2(maximum_frequency=max_frequency):
        for m in range(1, max_frequency + 1):
            mult = int(group.irrep(1,m).size // group.irrep(1,m).sum_of_squares_constituents)
            irreps.extend([group.irrep(1, m)] * mult)
    # ---- invariant map case: keep number of *output* channels constant across groups ----
    if last_layer:
        size = 0
        for irr in group.fibergroup.irreps():
            size += int(irr.size // irr.sum_of_squares_constituents)
        size = max(size, 1)
        print(f"New channel size {int(round(C/size))}")
        return max(1, int(round(C / size)))

    # ---- parameter-fix case: keep params roughly constant by compensating kernel basis dimension ----
    if layer_index > 1:
        # r_in and r_out are [trivial] + irreps
        if activation_type in ['fourier_relu', 'fourier_elu', 'norm_relu', 'norm_squash']:
            r_in  = FieldType(group, [group.trivial_repr] + list(irreps))
            r_out = FieldType(group, [group.trivial_repr] + list(irreps))
        elif activation_type in ['gated_shared_sigmoid']:
            r_in = FieldType(group, [group.trivial_repr] * 2 + irreps)
            r_out = FieldType(group, [group.trivial_repr] + irreps)
        elif activation_type in ['gated_sigmoid']:
            I = len(irreps)
            S = FieldType(group, irreps).size + 1
            M = S + I
            r_in = FieldType(group, [group.trivial_repr] * (I + 1) + irreps)
            r_out = FieldType(group, [group.trivial_repr] + irreps)
        tmp = R2Conv(
            r_in, r_out, kernel_size,
        )
        # basis dimension of constrained kernel space
        t = float(tmp.basisexpansion.dimension())

        # normalize w.r.t. the reference SFCNN factor you had:
        # 16 * s^2 * 3 / 4  (exactly as in your snippet)
        denom = 16.0 * (kernel_size ** 2) * 3.0 / 4.0
        t /= denom if denom > 0 else 1.0

        scale = 1.0 / max(np.sqrt(t), 1e-12)
        print(f"New channel size {int(round(C * scale))}")
        return max(1, int(round(C * scale)))

    # default: unchanged
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
            activation_type='gated_shared_sigmoid',
            max_frequency=max_frequency,
            layer_index=layer_idx,
            last_layer=(layer_idx == len(channels))
        )
        print(f"Layer {layer_idx}: nominal channels={C}, adjusted channels={adjusted_C}")