import pytest
import torch

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

def test_mnist_rot_dm_import_and_setup():
    # Just import and instantiate. The file has transforms and setup contract we can validate. :contentReference[oaicite:14]{index=14}
    from datasets_utils.data_classes import MnistRotDataModule
    dm = MnistRotDataModule(batch_size=4, data_dir=None)
    # We can’t rely on data files being present in CI; just ensure object exists and has attributes.
    assert hasattr(dm, "train_transform") and hasattr(dm, "test_transform")

def test_hnet_litmodule_integration():
    # Lightning plumbing at least instantiates and configures optimizers. :contentReference[oaicite:15]{index=15}
    from train import LitHnn
    m = LitHnn(n_classes=10, channels_per_block=(2,), layers_per_block=1, gated=True,
               kernel_size=3, invariant_channels=4, use_bn=False, non_linearity='n_relu', max_rot_order=1)
    opt_cfg = m.configure_optimizers()
    assert "optimizer" in opt_cfg and opt_cfg["optimizer"] is not None
