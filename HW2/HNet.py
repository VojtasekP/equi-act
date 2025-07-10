import os
from typing import Any

import torchmetrics
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import wandb
import torch

torch.set_num_threads(os.cpu_count())
torch.set_float32_matmul_precision('high')

from escnn import gspaces
from escnn import nn



import lightning as L
from pytorch_lightning.loggers import WandbLogger

from HW1.mnist_rot import MnistRotDataset, MnistRotDataModule

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# download the dataset
class HNet(torch.nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        max_rot_order = 2
        self.r2_act = gspaces.rot2dOnR2(-1, maximum_frequency=max_rot_order)
        CHANNELS_1 = 8
        CHANNELS_2 = 16
        CHANNELS_3 = 128
        FILTER_SIZE = 5
        PADDING_SIZE = (FILTER_SIZE-1) // 2

        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])

        self.input_type = in_type


        irreps_1 = CHANNELS_1 * [self.r2_act.irrep(m) for m in range(max_rot_order+1)]
        self.mask = nn.MaskModule(in_type, 29, margin=1)
        out_type_1 = nn.FieldType(self.r2_act, irreps_1)
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type_1, kernel_size=FILTER_SIZE, padding=PADDING_SIZE, bias=False),
            nn.NormNonLinearity(out_type_1),

            nn.R2Conv(out_type_1, out_type_1, kernel_size=FILTER_SIZE, padding=PADDING_SIZE, bias=False),
            nn.IIDBatchNorm2d(out_type_1),
        )

        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type_1, sigma=0.66, stride=2)
        )

        in_type = self.block1.out_type

        irreps_2 = CHANNELS_2 * [self.r2_act.irrep(m) for m in range(max_rot_order+1)]
        out_type_2 = nn.FieldType(self.r2_act, irreps_2)
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type_2, kernel_size=FILTER_SIZE, padding=PADDING_SIZE, bias=False),
            nn.NormNonLinearity(out_type_2),
            nn.R2Conv(out_type_2, out_type_2, kernel_size=FILTER_SIZE, padding=PADDING_SIZE, bias=False),
            nn.IIDBatchNorm2d(out_type_2),
        )

        self.pool2 = nn.PointwiseAvgPoolAntialiased(out_type_2, sigma=0.66, stride=2)



        in_type = self.block2.out_type

        irreps_3 = CHANNELS_3 * [self.r2_act.irrep(m) for m in range(max_rot_order+1)]
        out_type_3 = nn.FieldType(self.r2_act, irreps_3)

        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type_3, kernel_size=FILTER_SIZE, padding=PADDING_SIZE, bias=False),
            nn.NormNonLinearity(out_type_3),

            nn.R2Conv(out_type_3, out_type_3, kernel_size=FILTER_SIZE, padding=PADDING_SIZE, bias=False),
            nn.IIDBatchNorm2d(out_type_3),
        )


        c = 64
        output_invariant_type = nn.FieldType(self.r2_act, c*[self.r2_act.trivial_repr])
        self.invariant_map = nn.R2Conv(out_type_3, output_invariant_type, kernel_size=1, bias=False)
        self.fully_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(c*c),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(c*c, n_classes),
        )

    def forward(self, input: torch.Tensor):

        x = nn.GeometricTensor(input, self.input_type)

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        x = self.invariant_map(x)
        x = x.tensor

        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1))

        return x


class LitHnn(L.LightningModule):
    def __init__(self, n_classes=10):
        super().__init__()
        self.model = HNet(n_classes=n_classes)
        self.save_hyperparameters()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.lr = 1e-3

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)

    def on_load_checkpoint(self, checkpoint):
        # Clone tensors before loading to avoid memory location conflicts
        state_dict = checkpoint['state_dict']
        for key in list(state_dict.keys()):
            if '_basisexpansion.in_indices_' in key:
                state_dict[key] = state_dict[key].clone()


    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, self.train_acc)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_acc', acc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, self.val_acc)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, self.test_acc)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_acc', acc, prog_bar=True, logger=True)
        return loss

    def shared_step(self, batch, batch_idx, acc_metric):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)

        acc = acc_metric(y_hat, y)
        return loss, acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-5, weight_decay=1e-5)


if __name__ == '__main__':
    wandb_logger = WandbLogger(
        project='research_task_hws',
        name='hnet_rotation_invariant'
    )

    mnist_rot = MnistRotDataModule()
    g_cnn = LitHnn()

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints',
        filename='HNet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    trainer = Trainer(
        max_epochs=20,
        logger=wandb_logger,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min"), checkpoint_callback]
    )
    trainer.fit(g_cnn, datamodule=mnist_rot)

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        best_model = LitHnn.load_from_checkpoint(best_model_path)
        trainer.test(best_model, datamodule=mnist_rot)
    else:
        trainer.test(g_cnn, datamodule=mnist_rot)

    wandb.finish()

# TODO: Save the best model (using callbacks) and test it