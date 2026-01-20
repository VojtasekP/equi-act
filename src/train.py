import os
import json
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
from nets.RnNet import R2Net, R2PointNet
from nets.baseline_resnet import LitResNet18
from datasets_utils.data_classes import MnistRotDataModule, Resisc45DataModule, ColorectalHistDataModule, EuroSATDataModule, MnistRotPointCloudDataModule
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
torch.autograd.set_detect_anomaly(True)
class LitHnn(L.LightningModule):
    def __init__(self,
                 n_classes=10,
                 dataset: str = "mnist_rot",
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
                 residual: bool = False,
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
                 num_of_angles=32,
                 label_smoothing: float = 0.0,
                 ):
        super().__init__()

        self.save_hyperparameters()
        self.dataset = dataset
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
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
                          residual=residual,
                          activation_type=activation_type,
                          grey_scale=grey_scale,
                          img_size=img_size,
                          mnist=mnist
                          )

        # Clone basis index tensors to avoid overlapping storage issues when loading checkpoints.
        self._clone_in_indices()

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)

        self.eq_num_angles = num_of_angles
        self.invar_error_logging = invar_error_logging
        self.invar_check_every_n_epochs = max(1, int(invar_check_every_n_epochs))
        self.invar_chunk_size = max(1, int(invar_chunk_size))
        self.num_of_batches_to_use_for_invar_logging = max(1, int(num_of_batches_to_use_for_invar_logging))

    def _clone_in_indices(self):
        with torch.no_grad():
            for name, tensor in list(self.model.named_parameters()) + list(self.model.named_buffers()):
                if "in_indices_" not in name:
                    continue
                if not isinstance(tensor, torch.Tensor):
                    continue
                tensor.data = tensor.data.clone()

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
        # self._check_finite_params()
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        acc = acc_metric(y_hat, y)
        return loss, acc

    # def _check_finite_params(self):
    #     for name, p in self.model.named_parameters():
    #         if p is None:
    #             continue
    #         if not torch.isfinite(p).all():
    #             raise RuntimeError(f"NaN/Inf detected in parameter {name}, shape={tuple(p.shape)}")
    #         if p.grad is not None and not torch.isfinite(p.grad).all():
    #             raise RuntimeError(f"NaN/Inf detected in gradient of {name}, shape={tuple(p.grad.shape)}")

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, self.train_acc)
        # self._check_finite_params()
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
        if self.dataset == "mnist_rot":
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=self.burn_in_period),
                    optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.exp_dump),
                ],
                milestones=[self.burn_in_period],
            )
        else:
            cosine_iters = max(1, self.epochs - self.burn_in_period)
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            if self.burn_in_period > 0:
                scheduler = optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[
                        optim.lr_scheduler.LinearLR(
                            optimizer,
                            start_factor=0.1,
                            total_iters=self.burn_in_period,
                        ),
                        optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=cosine_iters,
                        ),
                    ],
                    milestones=[self.burn_in_period],
                )
            else:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=cosine_iters,
                    eta_min=self.lr * self.exp_dump,
                )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }


class LitHnnPointCloud(L.LightningModule):
    """
    Lightning module for R2PointNet (point cloud variant).
    Handles batches from PyG DataLoader with structure: batch.x, batch.pos, batch.edge_index, batch.y
    """
    def __init__(self,
                 n_classes=10,
                 dataset: str = "mnist_rot_p",
                 max_rot_order=2,
                 flip=False,
                 channels_per_block=(8, 16, 32),
                 kernels_per_block=(3, 3, 3),
                 paddings_per_block=(1, 1, 1),
                 conv_sigma=0.6,
                 pool_after_every_n_blocks=2,
                 activation_type="gated_sigmoid",
                 pool_size=2,
                 pool_sigma=0.66,
                 invar_type='norm',
                 pool_type='max',
                 invariant_channels=64,
                 bn="Normbn",
                 residual: bool = False,
                 lr=0.001,
                 weight_decay=0.0001,
                 grey_scale=False,
                 epochs=200,
                 burin_in_period=5,
                 exp_dump=0.9,
                 mnist=False,
                 label_smoothing: float = 0.0,
                 point_conv_n_rings: int = 3,
                 point_conv_frequencies_cutoff: float = 3.0):
        super().__init__()

        self.save_hyperparameters()
        self.dataset = dataset
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.burn_in_period = burin_in_period
        self.exp_dump = exp_dump

        # Use R2PointNet for point clouds
        self.model = R2PointNet(
            n_classes=n_classes,
            max_rot_order=max_rot_order,
            flip=flip,
            channels_per_block=channels_per_block,
            kernels_per_block=kernels_per_block,
            paddings_per_block=paddings_per_block,
            pool_after_every_n_blocks=pool_after_every_n_blocks,
            conv_sigma=conv_sigma,
            activation_type=activation_type,
            pool_size=pool_size,
            pool_sigma=pool_sigma,
            invar_type=invar_type,
            pool_type=pool_type,
            invariant_channels=invariant_channels,
            bn=bn,
            grey_scale=grey_scale,
            mnist=mnist,
            residual=residual,
            point_conv_n_rings=point_conv_n_rings,
            point_conv_frequencies_cutoff=point_conv_frequencies_cutoff
        )

        # Clone basis index tensors
        self._clone_in_indices()

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)

    def _clone_in_indices(self):
        with torch.no_grad():
            for name, tensor in list(self.model.named_parameters()) + list(self.model.named_buffers()):
                if "in_indices_" not in name:
                    continue
                if not isinstance(tensor, torch.Tensor):
                    continue
                tensor.data = tensor.data.clone()

    def on_fit_start(self):
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.model.parameters())

        self.logger.experiment.log({
            "model/num_trainable_params": n_params,
            "model/num_total_params": n_all
        })

    def shared_step(self, batch, acc_metric):
        """
        Handle PyG batch structure:
        - batch.x: features (total_points, feature_dim)
        - batch.pos: coordinates (total_points, 2)
        - batch.edge_index: connectivity (2, num_edges)
        - batch.y: labels (batch_size,)
        - batch.batch: batch assignment tensor (total_points,)
        """
        x = batch.x          # (total_points, 1)
        pos = batch.pos      # (total_points, 2)
        edge_index = batch.edge_index  # (2, num_edges)
        y = batch.y          # (batch_size,)
        batch_indices = batch.batch  # PyG's batch assignment tensor

        # Forward pass with batch info
        y_hat = self.model(x, pos, edge_index, batch=batch_indices)
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

    def test_step(self, batch, batch_idx):
        self.model.eval()
        loss, acc = self.shared_step(batch, self.test_acc)
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_acc',  acc,  prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if self.dataset == "mnist_rot_p":
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=self.burn_in_period),
                    optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.exp_dump),
                ],
                milestones=[self.burn_in_period],
            )
        else:
            # Fallback for other datasets
            cosine_iters = max(1, self.epochs - self.burn_in_period)
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            if self.burn_in_period > 0:
                scheduler = optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[
                        optim.lr_scheduler.LinearLR(
                            optimizer,
                            start_factor=0.1,
                            total_iters=self.burn_in_period,
                        ),
                        optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=cosine_iters,
                        ),
                    ],
                    milestones=[self.burn_in_period],
                )
            else:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=cosine_iters,
                    eta_min=self.lr * self.exp_dump,
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


def _parse_blocks(blocks_cfg):
    """
    Normalize blocks config into lists of channels, kernels, paddings.
    Accepts a Python list (e.g., from YAML) or a JSON/string payload like '[[16,7,3],[24,5,2]]'.
    """
    if blocks_cfg is None:
        return None
    parsed = None
    if isinstance(blocks_cfg, str):
        try:
            parsed = json.loads(blocks_cfg)
        except Exception:
            # fallback: split "16-7-3,24-5-2"
            try:
                parsed = []
                for item in blocks_cfg.split(","):
                    parts = item.strip().split("-")
                    if len(parts) != 3:
                        continue
                    parsed.append([int(parts[0]), int(parts[1]), int(parts[2])])
            except Exception:
                parsed = None
    else:
        parsed = blocks_cfg

    if not parsed:
        return None
    channels, kernels, paddings = [], [], []
    for triplet in parsed:
        if len(triplet) != 3:
            continue
        c, k, p = triplet
        channels.append(int(c))
        kernels.append(int(k))
        paddings.append(int(p))
    return (channels, kernels, paddings) if channels else None


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
    model_type = getattr(config, "model_type", "equivariant")
    train_subset_fraction = getattr(config, "train_subset_fraction", 1.0)
    model_export_dir = getattr(config, "model_export_dir", "saved_models")
    model_export_subdir = getattr(config, "model_export_subdir", default_subdir)
    model_export_subdir = _sanitize_component(model_export_subdir)
    if not model_export_subdir:
        model_export_subdir = default_subdir
    label_smoothing = float(getattr(config, "label_smoothing", 0.0))
    label_smoothing = max(0.0, min(label_smoothing, 0.999999))
    flip_flag = bool(getattr(config, "flip", False))
    baseline_pretrained = bool(getattr(config, "baseline_pretrained", False))
    blocks_cfg = _parse_blocks(getattr(config, "blocks", None))
    if blocks_cfg:
        base_channels_per_block, base_kernels_per_block, base_paddings_per_block = blocks_cfg
    else:
        base_channels_per_block = list(getattr(config, "channels_per_block", ()))
        base_kernels_per_block = list(getattr(config, "kernels_per_block", ()))
        base_paddings_per_block = list(getattr(config, "paddings_per_block", ()))
    mnist = False
    is_pointcloud = False  # Flag to detect point cloud mode

    if dataset == "mnist_rot":
        datamodule = MnistRotDataModule(batch_size=config.batch_size,
                                        data_dir="./src/datasets_utils/mnist_rotation_new",
                                        img_size=getattr(config, "img_size", None),
                                        train_fraction=train_subset_fraction,
                                        aug=config.aug,
                                        normalize=config.normalize)
        grey_scale = True
        mnist=True
    elif dataset == "mnist_rot_p":
        # Point cloud variant of MNIST
        k_neighbors = getattr(config, "k_neighbors", 8)
        radius = getattr(config, "radius", None)
        datamodule = MnistRotPointCloudDataModule(
            batch_size=config.batch_size,
            data_dir=getattr(config, "pointcloud_data_dir", None),
            train_fraction=train_subset_fraction,
            augment_rotation=config.aug,
            k_neighbors=k_neighbors,
            radius=radius,
            num_workers=getattr(config, "num_workers", None)
        )
        grey_scale = True
        mnist = True
        is_pointcloud = True
    elif dataset == "resisc45":
        datamodule = Resisc45DataModule(batch_size=config.batch_size,
                                        img_size=getattr(config, "img_size", None),
                                        train_fraction=train_subset_fraction,
                                        aug=config.aug,
                                        normalize=config.normalize)
        grey_scale = False
    elif dataset == "colorectal_hist":
        datamodule = ColorectalHistDataModule(batch_size=config.batch_size,
                                              img_size=getattr(config, "img_size", None),
                                              train_fraction=train_subset_fraction,
                                              aug=config.aug,
                                              normalize=config.normalize)
        grey_scale = False
    elif dataset == "eurosat":
        datamodule = EuroSATDataModule(batch_size=config.batch_size,
                                       img_size=getattr(config, "img_size", None),
                                       seed=seed,
                                       train_fraction=train_subset_fraction,
                                       aug=config.aug,
                                       normalize=config.normalize)
        grey_scale = False
        mnist = True  # Use MNIST channel scaling for consistent architecture
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    n_classes_dm = datamodule.num_classes
    img_size = getattr(datamodule, 'img_size', None)  # Point cloud datamodule doesn't have img_size
    wandb.config.update({
        "img_size": img_size,
        "dataset": dataset,
        "train_subset_fraction": train_subset_fraction,
        "seed": seed,
        "model_export_dir": model_export_dir,
        "model_export_subdir": model_export_subdir,
        "model_type": model_type,
        "label_smoothing": label_smoothing,
        "is_pointcloud": is_pointcloud,
    }, allow_val_change=True)

    if is_pointcloud:
        # Add point cloud specific config
        wandb.config.update({
            "k_neighbors": getattr(config, "k_neighbors", 8),
            "radius": getattr(config, "radius", None),
            "point_conv_n_rings": getattr(config, "point_conv_n_rings", 3),
            "point_conv_frequencies_cutoff": getattr(config, "point_conv_frequencies_cutoff", 3.0),
        }, allow_val_change=True)

    if model_type == "equivariant":
        wandb.config.update({
            "activation_type": getattr(config, "activation_type", None),
            "normalization": getattr(config, "bn", None),
            "blocks": blocks_cfg,
        }, allow_val_change=True)

        channels_per_block_updated = [int(c * config.channels_multiplier) for c in base_channels_per_block]
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

        # Choose model based on point cloud flag
        if is_pointcloud:
            # Use LitHnnPointCloud for point cloud data
            model = LitHnnPointCloud(
                n_classes=n_classes_dm,
                dataset=dataset,
                max_rot_order=config.max_rot_order,
                flip=config.flip,

                channels_per_block=channels_per_block_updated,
                kernels_per_block=base_kernels_per_block,
                paddings_per_block=base_paddings_per_block,
                pool_after_every_n_blocks=config.pool_after_every_n_blocks,
                conv_sigma=config.conv_sigma,
                activation_type=config.activation_type,

                pool_size=config.pool_size,
                pool_sigma=config.pool_sigma,
                invar_type=config.invar_type,
                pool_type=config.pool_type,
                invariant_channels=updated_invariant_channels,
                bn=config.bn,
                residual=getattr(config, "residual", False),
                lr=config.lr,
                weight_decay=config.weight_decay,
                epochs=config.epochs,
                burin_in_period=config.burin_in_period,
                exp_dump=config.exp_dump,

                grey_scale=grey_scale,
                mnist=mnist,
                label_smoothing=label_smoothing,
                point_conv_n_rings=getattr(config, "point_conv_n_rings", 3),
                point_conv_frequencies_cutoff=getattr(config, "point_conv_frequencies_cutoff", 3.0),
            )
        else:
            # Use LitHnn for grid-based data
            model = LitHnn(
                n_classes=n_classes_dm,
                dataset=dataset,
                max_rot_order=config.max_rot_order,
                flip=config.flip,

                channels_per_block=channels_per_block_updated,
                kernels_per_block=base_kernels_per_block,
                paddings_per_block=base_paddings_per_block,
                pool_after_every_n_blocks=config.pool_after_every_n_blocks,
                conv_sigma=config.conv_sigma,
                activation_type=config.activation_type,

                pool_size=config.pool_size,
                pool_sigma=config.pool_sigma,
                invar_type=config.invar_type,
                pool_type=config.pool_type,
                invariant_channels=updated_invariant_channels,
                bn=config.bn,
                residual=getattr(config, "residual", False),
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
                label_smoothing=label_smoothing,
            )
    elif model_type == "resnet18":
        in_channels = 1 if dataset == "mnist_rot" else 3
        wandb.config.update({
            "baseline_pretrained": baseline_pretrained,
            "in_channels": in_channels,
        }, allow_val_change=True)
        model = LitResNet18(
            n_classes=n_classes_dm,
            lr=config.lr,
            weight_decay=config.weight_decay,
            epochs=config.epochs,
            burin_in_period=config.burin_in_period,
            in_channels=in_channels,
            pretrained=baseline_pretrained,
            label_smoothing=label_smoothing,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    save_flag = bool(getattr(config, "save", True))
    chkpt = None
    if save_flag:
        chkpt = ModelCheckpoint(monitor='val_loss', filename='HNet-{epoch:02d}-{val_loss:.2f}',
                                save_top_k=1, mode='min')
    early = EarlyStopping(monitor='val_loss', mode='min', patience=config.patience, verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    callbacks = [early, lr_monitor]
    if chkpt is not None:
        callbacks.insert(0, chkpt)

    trainer = Trainer(
        max_epochs=config.epochs,
        logger=logger,
        callbacks=callbacks,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        strategy="auto",
        precision=config.precision,
        gradient_clip_val=1.0,
        deterministic=False,
        benchmark=False,
        log_every_n_steps=50,
    )

    trainer.fit(model, datamodule=datamodule)
    best_ckpt_path = chkpt.best_model_path if chkpt is not None else None
    best_val_loss = float(chkpt.best_model_score.item()) if chkpt is not None and chkpt.best_model_score is not None else None
    if best_ckpt_path:
        test_metrics = trainer.test(model=None, datamodule=datamodule, ckpt_path=best_ckpt_path)
    else:
        test_metrics = trainer.test(model=None, datamodule=datamodule)

    tm = test_metrics[0] if isinstance(test_metrics, list) and len(test_metrics) else {}
    test_acc = float(tm.get("test_acc", float("nan")))
    test_loss = float(tm.get("test_loss", float("nan")))

    exported_ckpt = ""
    if best_ckpt_path and save_flag:
        name_parts = [
            model_type,
            dataset,
            f"seed{seed}",
            "aug" if getattr(config, "aug", False) else "noaug",
        ]
        if model_type == "equivariant":
            name_parts.extend([
                getattr(config, "activation_type", None),
                getattr(config, "bn", None),
                "flip" if flip_flag else None,
                "residual" if getattr(config, "residual", False) else None,
            ])
        elif model_type == "resnet18":
            name_parts.extend([
                "resnet",
                "pretrained" if baseline_pretrained else "scratch"
            ])
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
    parser.add_argument("--dataset", type=str, default="mnist_rot",
                        choices=["mnist_rot", "mnist_rot_p", "resisc45", "colorectal_hist", "eurosat"],
                        help="Dataset to use. Use 'mnist_rot_p' for point cloud variant.")
    parser.add_argument("--model_type", type=str, default="equivariant",
                        choices=["equivariant", "resnet18"],
                        help="Choose between the equivariant model and a baseline ResNet18.")
    parser.add_argument("--baseline_pretrained", type=bool, default=False, help="Use ImageNet pretrained weights when model_type=resnet18.")
    parser.add_argument("--flip", type=bool, default=False, help="for O2")
    parser.add_argument("--aug", type=bool, default=True)
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument("--residual", type=bool, default=False, help="Enable residual connections between blocks.")
    parser.add_argument("--save", type=bool, default=True, help="If true, save best checkpoint and export copy.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing applied to cross-entropy loss.")
    parser.add_argument("--burin_in_period", type=int, default=10)
    parser.add_argument("--exp_dump", type=float, default=0.8)
    parser.add_argument("--channels_per_block", default=(16, 24, 32, 32, 48, 64))
    parser.add_argument("--kernels_per_block", default=(7, 5, 5, 5, 5, 5))
    parser.add_argument("--paddings_per_block", default=(3, 2, 2, 2, 2, 2))
    parser.add_argument("--blocks", type=str, default=None,
                        help='Optional JSON/list of [channels, kernel, padding] triplets; overrides *_per_block settings.')
    parser.add_argument("--channels_multiplier", type=float, default=1.0)
    parser.add_argument("--conv_sigma", type=float, default=0.6)
    parser.add_argument("--pool_after_every_n_blocks", default=2)
    parser.add_argument("--activation_type", default="norm_relu")
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

    # Point cloud specific arguments (only used when dataset="mnist_rot_p")
    parser.add_argument("--pointcloud_data_dir", type=str, default=None,
                        help="Directory containing point cloud .npz files (for mnist_rot_p dataset)")
    parser.add_argument("--k_neighbors", type=int, default=8,
                        help="Number of neighbors for kNN graph construction (point cloud mode)")
    parser.add_argument("--radius", type=float, default=None,
                        help="Radius for radius graph construction (if set, overrides k_neighbors in point cloud mode)")
    parser.add_argument("--point_conv_n_rings", type=int, default=3,
                        help="Number of concentric rings for R2PointConv bases (point cloud mode)")
    parser.add_argument("--point_conv_frequencies_cutoff", type=float, default=3.0,
                        help="Maximum circular harmonic frequency for R2PointConv (point cloud mode)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of dataloader workers (point cloud mode, defaults to cpu_count//2)")

    args = parser.parse_args()

    config = vars(args)
    train(config)
