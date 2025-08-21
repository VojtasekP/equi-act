import os
import torch
import torch.optim as optim
import torchmetrics
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import lightning as L
from lightning.pytorch.loggers import WandbLogger

import wandb
from HNet import HNet
from mnist_rot import MnistRotDataModule

torch.set_num_threads(os.cpu_count())
torch.set_float32_matmul_precision('high')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LitHnn(L.LightningModule):
    def __init__(self,
                 n_classes=10,
                 max_rot_order=2,
                 channels_per_block=(8, 16, 128),          # number of channels per block
                 layers_per_block=2,     # depth per block
                 gated=True,                     # switch between gated vs norm blocks
                 kernel_size=5,
                 pool_stride=2,
                 pool_sigma=0.66,
                 invariant_channels=64,
                 use_bn=True,
                 non_linearity='n_relu',
                 lr=0.001,
                 weight_decay=0.01,
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
                          gated=gated,
                          kernel_size=kernel_size,
                          pool_stride=pool_stride,
                          pool_sigma=pool_sigma,
                          invariant_channels=invariant_channels,
                          use_bn=use_bn,
                          non_linearity=non_linearity
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


# ---------------- Training Function ----------------
def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        logger = WandbLogger(project='HNet_gated', name='hnet')

        datamodule = MnistRotDataModule(batch_size=config.batch_size)

        model = LitHnn(
            n_classes=config.n_classes,
            max_rot_order=config.max_rot_order,
            channels_per_block=config.channels_per_block,
            layers_per_block=config.layers_per_block,  # depth per block
            gated=config.gated,  # switch between gated vs norm blocks
            kernel_size=config.kernel_size,
            pool_stride=config.pool_stride,
            pool_sigma=config.pool_sigma,
            invariant_channels=config.invariant_channels,
            use_bn=config.use_bn,
            non_linearity=config.non_linearity,
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        chkpt = ModelCheckpoint(monitor='val_loss', filename='HNet-{epoch:02d}-{val_loss:.2f}',
                               save_top_k=1, mode='min')

        early = EarlyStopping(monitor='val_loss', mode='min', patience=config.patience)

        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        trainer = Trainer(
            max_epochs=config.epochs,
            logger=logger,
            callbacks=[early, chkpt, lr_monitor]
        )

        trainer.fit(model, datamodule=datamodule)
        trainer.test(model=None, datamodule=datamodule, ckpt_path="best")


if __name__ == '__main__':
    train()

