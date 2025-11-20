import os
import shutil
from datetime import datetime
from pathlib import Path

import random
import numpy as np
import torch

import torch.optim as optim
import torchmetrics
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import lightning as L
from lightning.pytorch.loggers import WandbLogger

import wandb
from nets.RnNet import R2Net
from datasets_utils.data_classes import MnistRotDataModule, Resisc45DataModule, ColorectalHistDataModule, EuroSATDataModule
import argparse
import nets.equivariance_metric as em

torch.set_num_threads(os.cpu_count())
torch.set_float32_matmul_precision('high')
# list all available GPUs
print("Available GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
# use second GPU if available

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class LitHnn(L.LightningModule):
    def __init__(self,
                 n_classes=10,
                 max_rot_order=2,
                 flip=False,
                 channels_per_block=(8, 16, 32),          # number of channels per block
                 kernels_per_block=(3, 3, 3),           # kernel size per block
                 paddings_per_block=(1, 1, 1),          # padding per block
                 conv_sigma=0.6,
                 pool_after_every_n_blocks=2,     
                 pool_size=2,
                 pool_sigma=0.66,
                 invar_type='norm', 
                 pool_type='max',
                 invariant_channels=64,
                 bn="Normbn",
                 activation_type="gated_sigmoid",
                 lr=0.001,
                 weight_decay=0.0001,
                 grey_scale=False,
                 img_size=29,
                 epochs=200,
                 burin_in_period=5,
                 exp_dump=0.9,
                 mnist=False,
                 invar_error_logging=True,
                 invar_check_every_n_epochs=1,
                 invar_chunk_size=4,
                 num_of_batches_to_use_for_invar_logging=16,
                 num_of_angles=32
                 ):
        super().__init__()

        self.save_hyperparameters()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.burn_in_period = burin_in_period
        self.exp_dump = exp_dump
        self.model = R2Net(n_classes=n_classes,
                          max_rot_order=max_rot_order,
                          flip=flip,
                          channels_per_block=channels_per_block,
                          kernels_per_block=kernels_per_block,
                          paddings_per_block=paddings_per_block,
                          pool_after_every_n_blocks=pool_after_every_n_blocks,
                          conv_sigma=conv_sigma,
                          pool_size=pool_size,
                          pool_sigma=pool_sigma,
                          invar_type=invar_type,
                          pool_type=pool_type,
                          invariant_channels=invariant_channels,
                          bn=bn,
                          activation_type=activation_type,
                          grey_scale=grey_scale,
                          img_size=img_size,
                          mnist=mnist
                          )

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)

        self.eq_num_angles = num_of_angles
        self.invar_error_logging = invar_error_logging
        self.invar_check_every_n_epochs = max(1, int(invar_check_every_n_epochs))
        self.invar_chunk_size = max(1, int(invar_chunk_size))
        self.num_of_batches_to_use_for_invar_logging = max(1, int(num_of_batches_to_use_for_invar_logging))

    def _should_log_invar(self, stage: str, batch_idx: int) -> bool:
        if not self.invar_error_logging:
            return False
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return False
        if stage == "val":
            if getattr(trainer, "sanity_checking", False):
                return False
            if ((self.current_epoch + 1) % self.invar_check_every_n_epochs) != 0:
                return False
            num_batches = trainer.num_val_batches
            if isinstance(num_batches, (list, tuple)):
                num_batches = sum(int(n) for n in num_batches)
            num_batches = int(num_batches)
            if num_batches <= 0:
                return False
            if getattr(self, "_val_eq_epoch", None) != self.current_epoch:
                k = min(self.num_of_batches_to_use_for_invar_logging, num_batches)
                self._val_eq_epoch = self.current_epoch
                self._val_eq_batches = set(random.sample(range(num_batches), k))
            if batch_idx not in getattr(self, "_val_eq_batches", set()):
                return False
        elif stage == "test":
            
            return True
        else:
            return False
        return True

    def _compute_equivariant_error(self, x: torch.Tensor) -> float:
        with torch.no_grad():
            _, curve = em.check_equivariance_batch_r2(
                x,
                self.model,
                num_samples=self.eq_num_angles,
                chunk_size=self.invar_chunk_size,
            )
        return curve

    def on_fit_start(self):
        # count all trainable parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # optional: total including frozen ones
        n_all = sum(p.numel() for p in self.model.parameters())

        # log to W&B via Lightning's logger
        self.logger.experiment.log({
            "model/num_trainable_params": n_params,
            "model/num_total_params": n_all
        })

        # REMOVE OR CHANGE THIS LINE:
        # self.log("num_params", n_params, prog_bar=False, rank_zero_only=True)
        


    def shared_step(self, batch, acc_metric):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        acc = acc_metric(y_hat, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, self.train_acc)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_acc',  acc,  prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        loss, acc = self.shared_step(batch, self.val_acc)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_acc',  acc,  prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        if self._should_log_invar(stage="val", batch_idx=batch_idx):
            x, _ = batch
            eq_mean = self._compute_equivariant_error(x).mean()
            self.log('val_equi_error', eq_mean, prog_bar=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        self.model.eval()
        loss, acc = self.shared_step(batch, self.test_acc)
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_acc',  acc,  prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        if self._should_log_invar(stage="test", batch_idx=batch_idx):
            x, _ = batch
            eq_curve = self._compute_equivariant_error(x)
            eq_mean = eq_curve.mean()
            self.log('test_equi_error', eq_mean, prog_bar=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[
                        optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=self.burn_in_period),
                        optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.exp_dump)
                    ],
                    milestones=[self.burn_in_period]
                )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }


def _sanitize_component(value) -> str:
    component = str(value)
    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in component)
    sanitized = sanitized.strip("-_")
    return sanitized or "value"


def _copy_model_checkpoint(ckpt_path: str, export_dir: str, subdir: str, name_parts) -> str:
    if not ckpt_path:
        return ""

    export_dir_path = Path(export_dir) / subdir
    export_dir_path.mkdir(parents=True, exist_ok=True)

    base_name = "_".join(
        _sanitize_component(part) for part in name_parts if part not in (None, "")
    )
    if not base_name:
        base_name = f"model_{subdir}"

    dest_ckpt = export_dir_path / f"{base_name}.ckpt"
    shutil.copy2(ckpt_path, dest_ckpt)
    return str(dest_ckpt)



def _train_impl(config):
    print("Config:", config)
    default_subdir = datetime.utcnow().strftime("%Y%m%d")
    seed = getattr(config, "seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger = WandbLogger(name=config.project, project=config.project)

    dataset = getattr(config, "dataset", "None")
    train_subset_fraction = getattr(config, "train_subset_fraction", 1.0)
    model_export_dir = getattr(config, "model_export_dir", "saved_models")
    model_export_subdir = getattr(config, "model_export_subdir", default_subdir)
    model_export_subdir = _sanitize_component(model_export_subdir)
    if not model_export_subdir:
        model_export_subdir = default_subdir
    flip_flag = bool(getattr(config, "flip", False))
    mnist=False
    if dataset == "mnist_rot":
        datamodule = MnistRotDataModule(batch_size=config.batch_size, 
                                        data_dir="./src/datasets_utils/mnist_rotation_new", 
                                        img_size=getattr(config, "img_size", None),
                                        train_fraction=train_subset_fraction)
        grey_scale = True
        mnist=True
    elif dataset == "resisc45":
        datamodule = Resisc45DataModule(batch_size=config.batch_size,
                                        img_size=getattr(config, "img_size", None),
                                        train_fraction=train_subset_fraction)
        grey_scale = False
    elif dataset == "colorectal_hist":
        datamodule = ColorectalHistDataModule(batch_size=config.batch_size,
                                              img_size=getattr(config, "img_size", None),
                                              train_fraction=train_subset_fraction)
        grey_scale = False
    elif dataset == "eurosat":
        datamodule = EuroSATDataModule(batch_size=config.batch_size,
                                       img_size=getattr(config, "img_size", None),
                                       seed=seed,
                                       train_fraction=train_subset_fraction)
        grey_scale = False
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    n_classes_dm = datamodule.num_classes
    img_size = datamodule.img_size
    wandb.config.update({
        "img_size": img_size
    }, allow_val_change=True)
    # Log choices
    wandb.config.update({
        "dataset": dataset
    }, allow_val_change=True)
    wandb.config.update({
        "train_subset_fraction": train_subset_fraction
    }, allow_val_change=True)
    wandb.config.update({
        "activation_type": getattr(config, "activation_type", None),
        "normalization": getattr(config, "bn", None),
        "seed": seed,
        "model_export_dir": model_export_dir,
        "model_export_subdir": model_export_subdir,
    }, allow_val_change=True)
    channels_per_block_updated = [int(c * config.channels_multiplier) for c in config.channels_per_block]
    wandb.config.update({
        "channels_per_block": channels_per_block_updated
    }, allow_val_change=True)
    updated_invariant_channels = int(config.invariant_channels*config.channels_multiplier)
    wandb.config.update({
        'invariant_channels': updated_invariant_channels}
    , allow_val_change=True)
    invar_check_every = getattr(config, "invar_check_every_n_epochs", 1)
    invar_chunk_size = getattr(config, "invar_chunk_size", 4)
    wandb.config.update({
        'invar_check_every_n_epochs': invar_check_every,
        'invar_chunk_size': invar_chunk_size
    }, allow_val_change=True)
    
    model = LitHnn(
        n_classes=n_classes_dm,
        max_rot_order=config.max_rot_order,
        flip=config.flip,

        channels_per_block=channels_per_block_updated,
        kernels_per_block=config.kernels_per_block,
        paddings_per_block=config.paddings_per_block,
        pool_after_every_n_blocks=config.pool_after_every_n_blocks,
        conv_sigma=config.conv_sigma,
        activation_type=config.activation_type,

        pool_size=config.pool_size,
        pool_sigma=config.pool_sigma,
        invar_type=config.invar_type,
        pool_type=config.pool_type,
        invariant_channels=updated_invariant_channels,
        bn=config.bn,
        lr=config.lr,
        weight_decay=config.weight_decay,
        epochs=config.epochs,
        burin_in_period=config.burin_in_period,
        exp_dump=config.exp_dump,
        
        grey_scale=grey_scale,
        img_size=img_size,
        mnist=mnist,
        invar_error_logging=config.invar_error_logging,
        invar_check_every_n_epochs=invar_check_every,
        invar_chunk_size=invar_chunk_size,
        num_of_batches_to_use_for_invar_logging=config.num_of_batches_to_use_for_invar_logging,
        num_of_angles=config.num_of_angles,
    )

    chkpt = ModelCheckpoint(monitor='val_loss', filename='HNet-{epoch:02d}-{val_loss:.2f}',
                            save_top_k=1, mode='min')
    early = EarlyStopping(monitor='val_loss', mode='min', patience=config.patience, verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    trainer = Trainer(
        max_epochs=config.epochs,
        logger=logger,
        callbacks=[early, chkpt, lr_monitor],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        strategy="auto",
        precision=config.precision,
        deterministic=False,
        benchmark=False,
        log_every_n_steps=50,
    )

    trainer.fit(model, datamodule=datamodule)
    best_ckpt_path = chkpt.best_model_path or None
    best_val_loss = float(chkpt.best_model_score.item()) if chkpt.best_model_score is not None else None
    if best_ckpt_path:
        test_metrics = trainer.test(model=None, datamodule=datamodule, ckpt_path=best_ckpt_path)
    else:
        test_metrics = trainer.test(model=None, datamodule=datamodule)

    tm = test_metrics[0] if isinstance(test_metrics, list) and len(test_metrics) else {}
    test_acc = float(tm.get("test_acc", float("nan")))
    test_loss = float(tm.get("test_loss", float("nan")))

    exported_ckpt = ""
    if best_ckpt_path:
        name_parts = [
            dataset,
            getattr(config, "activation_type", None),
            getattr(config, "bn", None),
            "flip" if flip_flag else None,
            f"seed{seed}",
        ]
        exported_ckpt = _copy_model_checkpoint(best_ckpt_path, model_export_dir, model_export_subdir, name_parts)

    return {
        "best_val_loss": best_val_loss,
        "best_ckpt": best_ckpt_path,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "exported_ckpt": exported_ckpt,
    }


# --- THIS controls sweep vs. standalone ---
def train(config=None):
    if wandb.run is None:  # standalone run
        with wandb.init(config=config):
            cfg = wandb.config  # attr-style access
            _train_impl(cfg)
    else:  # already in a sweep
        cfg = wandb.config if config is None else config
        _train_impl(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="research_task_hws")
    parser.add_argument("--name", type=str, default="HNet_experiment")
    parser.add_argument("--dataset", type=str, default="resisc45",
                        choices=["mnist_rot", "resisc45", "colorectal_hist", "eurosat"])
    parser.add_argument("--flip", type=bool, default=False, help="for O2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--burin_in_period", type=int, default=15)
    parser.add_argument("--exp_dump", type=float, default=0.9)
    parser.add_argument("--channels_per_block", default=(16, 24, 32, 32, 48, 64))
    parser.add_argument("--kernels_per_block", default=(7, 5, 5, 5, 5, 5))
    parser.add_argument("--paddings_per_block", default=(1, 2, 2, 2, 2, 0))
    parser.add_argument("--channels_multiplier", type=float, default=1.0)
    parser.add_argument("--conv_sigma", type=float, default=0.6)
    parser.add_argument("--pool_after_every_n_blocks", default=2)
    parser.add_argument("--activation_type", default="gated_shared_sigmoid", choices=["gated_sigmoid","gated_shared_sigmoid", "norm_relu", "norm_squash", "fourier_relu_16", "fourier_elu_16", "fourier_relu_8", "fourier_elu_8", "fourier_relu_32", "fourier_elu_32", "fourier_relu_4", "fourier_elu_4", "non_equi_relu", "non_equi_bn"])
    parser.add_argument("--pool_size", type=int, default=2)
    parser.add_argument("--pool_sigma", type=float, default=0.66)
    parser.add_argument("--invar_type", type=str, default='norm', choices=['conv2triv', 'norm'])
    parser.add_argument("--pool_type", type=str, default='max', choices=['avg', 'max'])
    parser.add_argument("--invariant_channels", type=int, default=64)
    parser.add_argument("--bn", default="Normbn", choices=["IIDbn", "Normbn", "FieldNorm", "GNormBatchNorm"])
    parser.add_argument("--max_rot_order", type=float, default=3)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--invar_error_logging", type=bool, default=False , help="Disable invariance error logging to speed up training")
    parser.add_argument("--precision", type=str, default="32-true", choices=["16-mixed", "32-true"])
    parser.add_argument("--invar_check_every_n_epochs", type=int, default=1,
                        help="How often (in epochs) to run the invariance metric during validation.")
    parser.add_argument("--invar_chunk_size", type=int, default=4,
                        help="Number of rotated batches evaluated together when computing invariance error.")
    parser.add_argument("--num_of_batches_to_use_for_invar_logging", type=int, default=4,
                        help="Number of batches to use for invariance logging.")
    
    parser.add_argument("--num_of_angles", type=int, default=32,
                        help="Number of rotation angles to use when computing equivariance error.")
    
    parser.add_argument("--train_subset_fraction", type=float, default=1.0,
                        help="Fraction of the EuroSAT train split to use (0, 1].")

    parser.add_argument("--model_export_dir", type=str, default="saved_models",
                        help="Directory where the selected checkpoints will be copied for later analysis.")
    parser.add_argument("--model_export_subdir", type=str, default="",
                        help="Optional name of a sub-folder under model_export_dir shared across runs (defaults to current date).")
                        
    args = parser.parse_args()

    config = vars(args)
    train(config)
