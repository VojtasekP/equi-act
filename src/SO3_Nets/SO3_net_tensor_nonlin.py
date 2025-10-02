
import math
import torch
from escnn import gspaces, nn
from typing import Iterable, List, Sequence, Tuple, Union


def _broadcast_list(x: Union[int, Sequence[int]], n: int) -> List[int]:
    if isinstance(x, int):
        return [x] * n
    x = list(x)
    if len(x) == 1:
        return x * n
    assert len(x) == n, f"Expected length {n}, got {len(x)}"
    return x


def _uniform_type(act, l: int, channels: int) -> nn.FieldType:
    return nn.FieldType(act, [act.irrep(l)] * channels)


def make_tensorpoly_block_3d(
    act,
    in_type: nn.FieldType,
    irrep_l: int,
    channels: int,
    layers_num: int,
    degree_schedule: Union[int, Sequence[int]] = 1,
    kernel_size: int = 3,
    pad: int = 1,
    use_bn: bool = True,
) -> Tuple[nn.SequentialModule, nn.FieldType]:
    """
    Build a block that uses ONLY quadratic tensor-product modules as the nonlinearity.

    Args:
        act: escnn SO(3) gspace (rot3dOnR3).
        in_type: incoming FieldType.
        irrep_l: which irrep l (0,1,2,...) to use for the uniform stream.
        channels: number of copies of irrep(l) in the uniform stream.
        layers_num: number of layers inside the block.
        degree_schedule: int or list[int] of length layers_num. For layer i, we apply
            TensorProductModule 'degree_schedule[i]' times after the linear lift.
            IMPORTANT: stacking k tensor-product modules composes quadratic maps k times;
            the maximal polynomial degree grows roughly like 2**k.
        kernel_size, pad: conv hyperparams.
        use_bn: whether to add IIDBatchNorm3d after the conv lift.

    Returns:
        (block_module, out_type)
    """
    degs = _broadcast_list(degree_schedule, layers_num)

    layers = []
    cur_type = in_type

    for k in degs:
        # Lift to uniform type
        uni_type = _uniform_type(act, irrep_l, channels)
        layers.append(nn.R3Conv(cur_type, uni_type, kernel_size=kernel_size, padding=pad, bias=False))
        if use_bn:
            layers.append(nn.IIDBatchNorm3d(uniType:=uni_type))

        # Apply stacked quadratic tensor-product modules
        # Each module keeps the same type (uni_type -> uni_type)
        for _ in range(max(0, int(k))):
            layers.append(nn.TensorProductModule(uniType, uniType))

        cur_type = uni_type

    return nn.SequentialModule(*layers), cur_type


class TensorPolySONet(torch.nn.Module):
    """
    Pure tensor-polynomial SO(3)-equivariant 3D network.

    Key ideas:
      - All feature streams are mapped to a UNIFORM irrep(l) repeated 'channels' times.
      - Nonlinearity is realized by stacked TensorProductModule applications (quadratic each).
      - Degree control per-layer via 'degree_schedule' (stack count), noting degree grows ~2^k.

    Input:  [B, C_in, D, H, W]
    Output: class logits [B, n_classes]
    """
    def __init__(
        self,
        n_classes: int = 10,
        max_rot_order: int = 2,
        # architecture
        channels_per_block: Sequence[int] = (16, 32, 64),
        layers_per_block: int = 2,
        degrees_per_block: Union[int, Sequence[Union[int, Sequence[int]]]] = 1,
        # representation
        tensor_irrep_l: int = 1,
        # conv / pool
        kernel_size: int = 3,
        pool_stride: int = 2,
        pool_sigma: float = 0.66,
        use_bn: bool = True,
        # readout
        invariant_channels: int = 64,
        gray_scale: bool = True,
    ):
        super().__init__()

        # Group action: SO(3) on R^3
        self.r3_act = gspaces.rot3dOnR3(maximum_frequency=max_rot_order)

        in_channels = 1 if gray_scale else 3
        self.input_type = nn.FieldType(self.r3_act, in_channels * [self.r3_act.trivial_repr])

        self.blocks = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        pad = (kernel_size - 1) // 2

        if isinstance(degrees_per_block, int):
            block_degrees = [ [degrees_per_block] * layers_per_block for _ in channels_per_block ]
        else:
            tmp = list(degrees_per_block)
            if len(tmp) == 1 and not isinstance(tmp[0], (list, tuple)):
                block_degrees = [ [int(tmp[0])] * layers_per_block for _ in channels_per_block ]
            else:
                block_degrees = []
                for i, ch in enumerate(channels_per_block):
                    deg_i = tmp[i] if i < len(tmp) else tmp[-1]
                    if isinstance(deg_i, int):
                        block_degrees.append([deg_i] * layers_per_block)
                    else:
                        deg_i = list(deg_i)
                        assert len(deg_i) == layers_per_block, f"Block {i}: expected {layers_per_block} degrees, got {len(deg_i)}"
                        block_degrees.append([int(x) for x in deg_i])

        cur_type = self.input_type
        for i, (channels, deg_sched) in enumerate(zip(channels_per_block, block_degrees)):
            block, cur_type = make_tensorpoly_block_3d(
                act=self.r3_act,
                in_type=cur_type,
                irrep_l=tensor_irrep_l,
                channels=channels,
                layers_num=layers_per_block,
                degree_schedule=deg_sched,
                kernel_size=kernel_size,
                pad=pad,
                use_bn=use_bn,
            )
            self.blocks.append(block)
            self.pools.append(nn.PointwiseAvgPoolAntialiased(cur_type, sigma=pool_sigma, stride=pool_stride))

        # Invariant projection and head
        c = invariant_channels
        self.inv_type = nn.FieldType(self.r3_act, c * [self.r3_act.trivial_repr])
        self.invariant_map = nn.R3Conv(cur_type, self.inv_type, kernel_size=1, bias=False)

        self.avg = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(c),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(c, n_classes),
        )

    def forward(self, input: torch.Tensor):
        x = nn.GeometricTensor(input, self.input_type)
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            x = pool(x)
        x = self.invariant_map(x)
        x = self.avg(x.tensor).flatten(1)
        return self.head(x)
