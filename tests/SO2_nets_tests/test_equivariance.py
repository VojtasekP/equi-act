import torch
import pytest
from escnn import nn
from nets.equivariance_metric import chech_equivariance_batch_r2, chech_invariance_batch_r2



import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class TinyEquivariantCNN(torch.nn.Module):
    def __init__(self, r2_act, input_type, hidden_fields=2):
        super().__init__()
        self.input_type = input_type
        self.r2_act = r2_act
        out_type = nn.FieldType(r2_act, [r2_act.irrep(0), r2_act.irrep(1), r2_act.irrep(2), r2_act.irrep(3)] * hidden_fields)

        self.layers = nn.SequentialModule(
            nn.R2Conv(self.input_type, out_type, kernel_size=3, padding=1, bias=True),
        )
        self.output_type = self.layers.out_type

    def forward_features(self, x):
        gx = x if isinstance(x, nn.GeometricTensor) else nn.GeometricTensor(x, self.input_type)
        return self.layers(gx)
    
def test_check_equivariance_batch_r2_two_layer_model(random_batch, r2_act, input_type_gray):
    model = TinyEquivariantCNN(r2_act, input_type_gray)
    num_samples = 16
    thetas, errs = chech_equivariance_batch_r2(
        random_batch, model, num_samples=num_samples, chunk_size=2
    )

    assert len(thetas) == num_samples
    assert errs.shape == (num_samples,)
    log.debug("thetas %s errs %s", thetas, errs.tolist())
    log.debug("avg error: %f", errs.mean())

def test_check_invariance_batch_r2_two_layer_model(random_batch, r2_act, input_type_gray):
    model = TinyEquivariantCNN(r2_act, input_type_gray)
    num_samples = 16
    thetas, errs = chech_invariance_batch_r2(
        random_batch, model, num_samples=num_samples, chunk_size=2
    )

    assert len(thetas) == num_samples
    assert errs.shape == (num_samples,)
    log.debug("thetas %s errs %s", thetas, errs.tolist())
    log.debug("avg error: %f", errs.mean())
