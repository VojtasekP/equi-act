import torch
from escnn import gspaces, nn

def norm_nonlinearity(in_type):
    return nn.NormNonLinearity(in_type=in_type, )


class EquivariantConvBlock(nn.EquivariantModule):
    def __init__(self, in_type, representation, channels, kenrel_size, padding, activation, batchnorm):
        pass
    def forward():
        pass








