
from escnn.group import *
from escnn.gspaces import *
from escnn.nn import FieldType
from escnn.nn import GeometricTensor
from escnn.nn import R2Conv
from escnn.nn.modules.equivariant_module import EquivariantModule
from escnn.nn import SequentialModule
import torch
import torch.nn.functional as F

from typing import List, Tuple, Any

import numpy as np



def _build_kernel(G: Group, irrep: List[tuple]):
    kernel = []
    
    for irr in irrep:
        irr = G.irrep(*irr)
        
        c = int(irr.size//irr.sum_of_squares_constituents)
        k = irr(G.identity)[:, :c] * np.sqrt(irr.size)
        kernel.append(k.T.reshape(-1))
    
    kernel = np.concatenate(kernel)
    return kernel
    

class AdaptiveFourierPointwise(EquivariantModule):
    
    def __init__(
            self,
            gspace: GSpace,
            channels: int,
            irreps: List,
            input_type_irreps: FieldType = None,
            num_grid_points: int = 16,
            function: str = 'p_relu',
            inplace: bool = True,
            out_irreps: List = None,
            normalize: bool = True,
            **grid_kwargs
    ):

        assert isinstance(gspace, GSpace)
        
        super(AdaptiveFourierPointwise, self).__init__()

        self.space = gspace
        
        G: Group = gspace.fibergroup
        
        self.rho = G.spectral_regular_representation(*irreps, name=None)

        self.in_type = FieldType(self.space, [self.rho] * channels)

        if out_irreps is None:
            # the representation in input is preserved
            self.out_type = self.in_type
            self.rho_out = self.rho
        else:
            self.rho_out = G.spectral_regular_representation(*out_irreps, name=None)
            self.out_type = FieldType(self.space, [self.rho_out] * channels)

        # retrieve the activation function to apply
        if function == 'p_relu':
            self._function = F.relu_ if inplace else F.relu
        elif function == 'p_elu':
            self._function = F.elu_ if inplace else F.elu
        elif function == 'p_sigmoid':
            self._function = torch.sigmoid_ if inplace else F.sigmoid
        elif function == 'p_tanh':
            self._function = torch.tanh_ if inplace else F.tanh
        else:
            raise ValueError('Function "{}" not recognized!'.format(function))
        
        self.kernel = _build_kernel(G, irreps)
        assert self.kernel.shape[0] == self.rho.size

        if normalize:
            self.kernel = self.kernel / np.linalg.norm(self.kernel)
        self.kernel = self.kernel.reshape(-1, 1)
        
        # grid = G.grid(*grid_args, **grid_kwargs)
        



        output_type_irreps = FieldType(self.space, [self.rho] * channels)
        self.sampling_branch = SequentialModule(
            R2Conv(input_type=input_type_irreps, out_type=input_type_irreps, kernel_size=1, bias=False),
        )

    def generate_A_Ainv(self, angles):

        grid = [self.space.fibergroup.element(float(theta)) for theta in angles]
        A = np.concatenate(
            [
                self.rho(g) @ self.kernel
                for g in grid
            ], axis=1
        ).T

        A_out = A
        rho_out_extended = self.rho_out

        eps = 1e-8
        Ainv = np.linalg.inv(A_out.T @ A_out + eps * np.eye(rho_out_extended.size)) @ A_out.T


        self.register_buffer('A', torch.tensor(A, dtype=torch.get_default_dtype()))
        self.register_buffer('Ainv', torch.tensor(Ainv, dtype=torch.get_default_dtype()))

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""

        Applies the pointwise activation function on the input fields

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map after the non-linearities have been applied

        """

        assert input.type == self.in_type
        
        shape = input.shape
        x_hat = input.tensor.view(shape[0], len(self.in_type), self.rho.size, *shape[2:])
        thetas = self.grid_branch(input.tensor)
        self.generate_A_Ainv(thetas)

        x = torch.einsum('bcf...,gf->bcg...', x_hat, self.A)
        
        y = self._function(x)

        y_hat = torch.einsum('bcg...,fg->bcf...', y, self.Ainv)

        y_hat = y_hat.reshape(shape[0], self.out_type.size, *shape[2:])

        return GeometricTensor(y_hat, self.out_type, input.coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:

        assert len(input_shape) >= 2
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        spatial_shape = input_shape[2:]

        return (b, self.out_type.size, *spatial_shape)

    def check_equivariance(self, atol: float = 1e-5, rtol: float = 2e-2, assert_raise: bool = True) -> List[Tuple[Any, float]]:
    
        c = self.in_type.size
        B = 128
        x = torch.randn(B, c, *[3]*self.space.dimensionality)

        # since we mostly use non-linearities like relu or eu,l we make sure the average value of the features is
        # positive, such that, when we test inputs with only frequency 0 (or only low frequencies), the output is not
        # zero everywhere
        x = x.view(B, len(self.in_type), self.rho.size, *[3]*self.space.dimensionality)
        p = 0
        for irr in self.rho.irreps:
            irr = self.space.irrep(*irr)
            if irr.is_trivial():
                x[:, :, p] = x[:, :, p].abs()
            p+=irr.size

        x = x.view(B, self.in_type.size, *[3]*self.space.dimensionality)

        errors = []

        # for el in self.space.testing_elements:
        for _ in range(100):
            
            el = self.space.fibergroup.sample()
    
            x1 = GeometricTensor(x.clone(), self.in_type)
            x2 = GeometricTensor(x.clone(), self.in_type).transform_fibers(el)

            out1 = self(x1).transform_fibers(el)
            out2 = self(x2)

            out1 = out1.tensor.view(B, len(self.out_type), self.rho_out.size, *out1.shape[2:]).detach().numpy()
            out2 = out2.tensor.view(B, len(self.out_type), self.rho_out.size, *out2.shape[2:]).detach().numpy()

            errs = np.linalg.norm(out1 - out2, axis=2).reshape(-1)
            errs[errs < atol] = 0.
            norm = np.sqrt(np.linalg.norm(out1, axis=2).reshape(-1) * np.linalg.norm(out2, axis=2).reshape(-1))
            
            relerr = errs / norm

            # print(el, errs.max(), errs.mean(), relerr.max(), relerr.min())

            if assert_raise:
                assert relerr.mean()+ relerr.std() < rtol, \
                    'The error found during equivariance check with element "{}" is too high: max = {}, mean = {}, std ={}' \
                        .format(el, relerr.max(), relerr.mean(), relerr.std())

            # errors.append((el, errs.mean()))
            errors.append(relerr)

        # return errors
        return np.concatenate(errors).reshape(-1)

