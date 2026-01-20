import torch
import torch.optim as optim
import torchmetrics
import lightning as L
from torch import nn
from torchvision import models


class LitResNet18(L.LightningModule):
    def __init__(
        self,
        n_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        burin_in_period: int = 5,
        in_channels: int = 3,
        pretrained: bool = False,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.burn_in_period = burin_in_period

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        if in_channels != 3:
            conv1 = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels,
                conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=False,
            )
            if weights is not None:
                with torch.no_grad():
                    new_w = conv1.weight.mean(dim=1, keepdim=True)
                    if in_channels > 1:
                        new_w = new_w.repeat(1, in_channels, 1, 1)
                    backbone.conv1.weight.copy_(new_w)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, n_classes)
        self.model = backbone

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)

    def on_fit_start(self):
        # Log parameter counts for visibility in W&B
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.model.parameters())
        if self.logger is not None and hasattr(self.logger, "experiment"):
            self.logger.experiment.log({
                "model/num_trainable_params": n_params,
                "model/num_total_params": n_all,
            })

    def forward(self, x):
        return self.model(x)

    def forward_invar_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features after avgpool, before final FC layer.

        Returns 512-dimensional feature vector (not classification logits).
        """
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)  # (B, 512)

        return x

    def shared_step(self, batch, acc_metric):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        acc = acc_metric(logits, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, self.train_acc)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, self.val_acc)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, self.test_acc)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # Linear warmup into cosine decay.
        warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=max(1, self.burn_in_period),
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.epochs - self.burn_in_period),
            eta_min=0.0,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.burn_in_period],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"},
        }
