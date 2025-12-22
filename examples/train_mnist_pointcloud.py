"""
Example script for training R2PointNet on MNIST rotation point clouds.

This demonstrates the complete workflow:
1. Load point cloud data
2. Create R2PointNet model
3. Train with Lightning

Usage:
    # First, convert MNIST to point clouds
    python src/datasets_utils/mnist_to_pointcloud.py

    # Then run training
    python examples/train_mnist_pointcloud.py --max_epochs 10
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import torch
import argparse
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import lightning as L

from nets.RnNet import R2PointNet
from datasets_utils.data_classes import MnistRotPointCloudDataModule


class LitR2PointNet(L.LightningModule):
    """Lightning wrapper for R2PointNet"""

    def __init__(self,
                 n_classes=10,
                 max_rot_order=4,
                 channels_per_block=(8, 16, 32),
                 activation_type="gated_sigmoid",
                 bn="IIDbn",
                 lr=0.001,
                 weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = R2PointNet(
            n_classes=n_classes,
            max_rot_order=max_rot_order,
            channels_per_block=channels_per_block,
            activation_type=activation_type,
            bn=bn,
            grey_scale=True,  # MNIST is grayscale
            mnist=True
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = torch.nn.ModuleList([])
        self.val_acc = torch.nn.ModuleList([])

    def forward(self, x, coords, edge_index):
        return self.model(x, coords, edge_index)

    def training_step(self, batch, batch_idx):
        # batch is a PyG Batch object
        x = batch.x          # (total_points, 1)
        pos = batch.pos      # (total_points, 2)
        edge_index = batch.edge_index  # (2, num_edges)
        y = batch.y          # (batch_size,)

        # Forward pass
        logits = self(x, pos, edge_index)
        loss = self.loss_fn(logits, y)

        # Metrics
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch.x
        pos = batch.pos
        edge_index = batch.edge_index
        y = batch.y

        logits = self(x, pos, edge_index)
        loss = self.loss_fn(logits, y)

        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x = batch.x
        pos = batch.pos
        edge_index = batch.edge_index
        y = batch.y

        logits = self(x, pos, edge_index)
        loss = self.loss_fn(logits, y)

        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log('test_loss', loss)
        self.log('test_acc', acc)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Train R2PointNet on MNIST point clouds")

    # Model args
    parser.add_argument('--max_rot_order', type=int, default=4,
                       help='Maximum rotation order')
    parser.add_argument('--channels', type=int, nargs='+', default=[8, 16, 32],
                       help='Channels per block')
    parser.add_argument('--activation_type', type=str, default='gated_sigmoid',
                       choices=['gated_sigmoid', 'norm_relu', 'fourier_relu_16'],
                       help='Activation type')
    parser.add_argument('--bn', type=str, default='IIDbn',
                       choices=['IIDbn', 'Normbn', 'FieldNorm'],
                       help='Batch normalization type')

    # Data args
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--k_neighbors', type=int, default=8,
                       help='Number of neighbors for kNN graph')
    parser.add_argument('--augment_rotation', action='store_true',
                       help='Enable rotation augmentation')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Point cloud data directory')

    # Training args
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=50,
                       help='Maximum epochs')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Dataloader workers')

    # Logging args
    parser.add_argument('--log_dir', type=str, default='lightning_logs',
                       help='Log directory')
    parser.add_argument('--experiment_name', type=str, default='mnist_pointcloud',
                       help='Experiment name')

    args = parser.parse_args()

    print("=" * 60)
    print("Training R2PointNet on MNIST Point Clouds")
    print("=" * 60)
    print(f"Max rotation order: {args.max_rot_order}")
    print(f"Channels: {args.channels}")
    print(f"Activation: {args.activation_type}")
    print(f"Batch norm: {args.bn}")
    print(f"Batch size: {args.batch_size}")
    print(f"kNN: k={args.k_neighbors}")
    print(f"Rotation aug: {args.augment_rotation}")
    print("=" * 60)

    # Create datamodule
    dm = MnistRotPointCloudDataModule(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        augment_rotation=args.augment_rotation,
        k_neighbors=args.k_neighbors,
        num_workers=args.num_workers
    )

    # Prepare and setup data
    try:
        dm.prepare_data()
        dm.setup()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease run the conversion script first:")
        print("  python src/datasets_utils/mnist_to_pointcloud.py")
        return

    # Create model
    model = LitR2PointNet(
        n_classes=10,
        max_rot_order=args.max_rot_order,
        channels_per_block=tuple(args.channels),
        activation_type=args.activation_type,
        bn=args.bn,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=3,
        filename='pointnet-{epoch:02d}-{val_acc:.4f}'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )

    # Logger
    logger = CSVLogger(
        save_dir=args.log_dir,
        name=args.experiment_name
    )

    # Trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=10,
        enable_progress_bar=True
    )

    # Train
    print("\nStarting training...")
    trainer.fit(model, dm)

    # Test
    print("\nRunning test...")
    trainer.test(model, dm)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Logs saved to: {logger.log_dir}")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
