import os
from typing import List

import torch
import torch.optim as optim
import torchmetrics
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import lightning as L
from lightning.pytorch.loggers import WandbLogger

import wandb
from SO2_Nets.HNet import HNet
from datasets_utils.data_classes import MnistRotDataModule, Resisc45DataModule, ColorectalHistDataModule
import argparse

torch.set_num_threads(os.cpu_count())
torch.set_float32_matmul_precision('high')
# list all available GPUs
print("Available GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
# use second GPU if available

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)


class LitHnn(L.LightningModule):
    def __init__(self,
                 n_classes=10,
                 max_rot_order=2,
                 channels_per_block=(8, 16, 128),          # number of channels per block
                 layers_per_block=2,     
                 kernel_size=5,
                 pool_stride=2,
                 pool_sigma=0.66,
                 invariant_channels=64,
                 bn=True,
                 activation_type="gated_relu",
                 lr=0.001,
                 weight_decay=0.01,
                 grey_scale=False
                 ):
        super().__init__()

        self.save_hyperparameters()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = HNet(n_classes=n_classes,
                          max_rot_order=max_rot_order,
                          channels_per_block=channels_per_block,
                          layers_per_block=layers_per_block,
                          kernel_size=kernel_size,
                          pool_stride=pool_stride,
                          pool_sigma=pool_sigma,
                          invariant_channels=invariant_channels,
                          bn=bn,
                          activation_type=activation_type,
                          grey_scale=grey_scale
                          )

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)

    def shared_step(self, batch, acc_metric):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        acc = acc_metric(y_hat, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, self.train_acc)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc',  acc,  prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, self.val_acc)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc',  acc,  prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, self.test_acc)
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_acc',  acc,  prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, )

        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }



def _train_impl(config):
    print("Config:", config)
    logger = WandbLogger(project=config.project, name=config.name)

    dataset = getattr(config, "dataset", "None")

    if dataset == "mnist_rot":
        datamodule = MnistRotDataModule(batch_size=config.batch_size, data_dir="./src/datasets_utils/mnist_rotation_new")
        n_classes_dm = getattr(datamodule, "num_classes", getattr(config, "n_classes", None))
    elif dataset == "resisc45":
        datamodule = Resisc45DataModule(batch_size=config.batch_size,
                                        img_size=getattr(config, "img_size", 128))
        n_classes_dm = datamodule.num_classes
    elif dataset == "colorectal_hist":
        datamodule = ColorectalHistDataModule(batch_size=config.batch_size,
                                              img_size=getattr(config, "img_size", 128))
        n_classes_dm = datamodule.num_classes
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Log choices
    wandb.config.update({
        "dataset": dataset
    }, allow_val_change=True)

    model = LitHnn(
        n_classes=int(n_classes_dm),
        max_rot_order=config.max_rot_order,
        channels_per_block=config.channels_per_block,
        layers_per_block=config.layers_per_block,
        activation_type=config.activation_type,
        kernel_size=config.kernel_size,
        pool_stride=config.pool_stride,
        pool_sigma=config.pool_sigma,
        invariant_channels=config.invariant_channels,
        bn=config.bn,
        lr=config.lr,
        weight_decay=config.weight_decay,
        grey_scale=config.gray_scale
    )

    chkpt = ModelCheckpoint(monitor='val_loss', filename='HNet-{epoch:02d}-{val_loss:.2f}',
                            save_top_k=1, mode='min')
    early = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=True, min_delta=0.001)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
        max_epochs=config.epochs,
        logger=logger,
        callbacks=[early, chkpt, lr_monitor],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        deterministic=False,
        benchmark=True
    )

    trainer.fit(model, datamodule=datamodule)
    best_ckpt_path = chkpt.best_model_path or None
    best_val_loss = float(chkpt.best_model_score.item()) if chkpt.best_model_score is not None else None

    if best_ckpt_path:
        test_metrics = trainer.test(model=None, datamodule=datamodule, ckpt_path=best_ckpt_path)
    else:
        test_metrics = trainer.test(model=None, datamodule=datamodule)

    tm = test_metrics[0] if isinstance(test_metrics, list) and len(test_metrics) else {}

    return {
        "best_val_loss": best_val_loss,
        "best_ckpt": best_ckpt_path,
        "test_acc": float(tm.get("test_acc", float("nan"))),
        "test_loss": float(tm.get("test_loss", float("nan")))
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
                        choices=["mnist_rot", "resisc45", "colorectal_hist"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--channels_per_block", default=[16, 32, 64])
    parser.add_argument("--layers_per_block", default=2)
    parser.add_argument("--activation_type", default="gated_relu", choices=["gated_relu","norm_relu", "norm_squash", "pointwise_relu","fourier_relu", "fourier_elu"])
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--pool_stride", type=int, default=2)
    parser.add_argument("--pool_sigma", type=float, default=1.0)
    parser.add_argument("--invariant_channels", type=int, default=64)
    parser.add_argument("--bn", default="IIDbn", choices=["IIDbn", "Normbn", "FieldNorm", "GNormBatchNorm"])
    parser.add_argument("--max_rot_order", type=float, default=3)
    parser.add_argument("--img_size", type=int, default=28)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--gray_scale", type=bool, default=False)
    args = parser.parse_args()

    config = vars(args)
    train(config)
