import os
from typing import Tuple

import lightning as L
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset, ClassLabel
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Pad, Resize, RandomRotation, InterpolationMode, ToTensor, CenterCrop, \
    RandomResizedCrop

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MnistRotDataset(Dataset):

    def __init__(self, mode, transform=None):
        assert mode in ['train', 'test']

        if mode == "train":
            file = "/mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
        else:
            file = "/mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"

        self.transform = transform

        data = np.loadtxt(file, delimiter=' ')

        self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image, mode='F')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)


class MnistRotDataModule(L.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

        self.train_transform = Compose([
            Pad((0, 0, 1, 1), fill=0),  # Pad the image
            Resize(87),  # Upscale
            Resize(29),  # Downscale back
            ToTensor(),  # Convert to tensor
        ])

        # Validation & Test Transforms (no rotation for consistency)
        self.test_transform = Compose([
            Pad((0, 0, 1, 1), fill=0),
            Resize(87),
            Resize(29),
            ToTensor(),
        ])

    def setup(self, stage: str):
        if stage == 'fit':
            self.mnist_full = MnistRotDataset(mode='train', transform=self.train_transform)
            self.mnist_train, self.mnist_val = torch.utils.data.random_split(self.mnist_full,
                                                                             [10000, 2000],
                                                                             generator=torch.Generator().manual_seed(
                                                                                 24))
        if stage == 'test':
            self.mnist_test = MnistRotDataset(mode='test', transform=self.test_transform)
        if stage == 'predict':
            self.mnist_predict = MnistRotDataset(mode='test', transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() // 2,  # Use multiple workers
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() // 2,  # Use multiple workers
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=True  # Keep workers alive between epochs
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() // 2,  # Use multiple workers
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=True  # Keep workers alive between epochs
        )

    def predict_dataloader(self):
        return DataLoader(
            self.mnist_predict,
            batch_size=self.batch_size,
            num_workers=os.cpu_count() // 2,  # Use multiple workers
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=True  # Keep workers alive between epochs
        )


class HFImageTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, transform, image_key: str = "image", label_key: str = "label"):
        self.ds = hf_split
        self.tf = transform
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex[self.image_key]  # PIL.Image (decoded by 🤗 Datasets)
        y = int(ex[self.label_key])
        return self.tf(img), y


# ---------- common transforms (RGB) ----------
def make_transforms(img_size: int) -> Tuple[Compose, Compose]:

    train_tf = Compose([
        RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        ToTensor()
    ])

    eval_tf = Compose([
        Resize(img_size),
        CenterCrop(img_size),
        ToTensor(),
    ])

    return train_tf, eval_tf

class Resisc45DataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 256, img_size: int = 150):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = 45
        max_image_size = 256
        if img_size > max_image_size:
            img_size = max_image_size
        self.img_size = img_size
        self.train_tf, self.eval_tf = make_transforms(img_size)

    def setup(self, stage=None):

        ds = load_dataset("timm/resisc45")

        train_split = ds["train"]
        val_split = ds["validation"]
        test_split = ds["test"]

        self.train_ds = HFImageTorchDataset(train_split, self.train_tf, "image", "label")
        self.val_ds = HFImageTorchDataset(val_split, self.eval_tf,  "image", "label")
        self.test_ds = HFImageTorchDataset(test_split, self.eval_tf,  "image", "label")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=os.cpu_count() // 2, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=os.cpu_count() // 2, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=os.cpu_count() // 2, pin_memory=True)


# ---------- Colorectal Histology: stratified 85/7.5/7.5 from TRAIN(5000) ----------
class ColorectalHistDataModule(L.LightningDataModule):
    def __init__(self,
                 batch_size: int = 64,
                 img_size: int = 150,
                 seed: int = 42):
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_classes = 8
        max_image_size = 150
        if img_size > max_image_size:
            img_size = max_image_size
        self.img_size = img_size
        self.train_tf, self.eval_tf = make_transforms(img_size)


    def setup(self, stage=None):
        ds = load_dataset("dpdl-benchmark/colorectal_histology")
        full = ds["train"]

        # Cast label column to ClassLabel
        num_classes = 8
        full = full.cast_column("label", ClassLabel(num_classes=num_classes))

        # 85% train / 15% temp
        split1 = full.train_test_split(test_size=0.15, seed=self.seed, stratify_by_column="label")
        train_split = split1["train"]
        temp_split = split1["test"]

        # split temp into val/test 50/50 => 7.5% each
        split2 = temp_split.train_test_split(test_size=0.5, seed=self.seed, stratify_by_column="label")
        val_split = split2["train"]
        test_split = split2["test"]

        self.train_ds = HFImageTorchDataset(train_split, self.train_tf, "image", "label")
        self.val_ds = HFImageTorchDataset(val_split, self.eval_tf, "image", "label")
        self.test_ds = HFImageTorchDataset(test_split, self.eval_tf, "image", "label")

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=os.cpu_count() // 2, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=os.cpu_count() // 2, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=os.cpu_count() // 2, pin_memory=True)
