import os
from typing import Tuple
from pathlib import Path
import lightning as L
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset, ClassLabel
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, random_split
from torchvision.transforms import Compose, Pad, Resize, RandomRotation, InterpolationMode, ToTensor, RandomVerticalFlip, RandomHorizontalFlip 
from torch_geometric.transforms import NormalizeScale, SamplePoints
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader as PyGDataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MnistRotDataset(Dataset):

    def __init__(self, mode, transform=None, data_dir=None):
        assert mode in ['train', 'test']

        self.transform = transform
        root = Path(data_dir) if data_dir else (Path(__file__).resolve().parents[1] / "SO2_Nets" / "datasets_utils" / "mnist_rotation_new")
        if mode == 'train':
            file = root / "mnist_all_rotation_normalized_float_train_valid.amat"
        elif mode == 'test':
            file = root / "mnist_all_rotation_normalized_float_test.amat"
        else:
            raise ValueError("mode must be 'train' or 'test'")
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
    def __init__(self, batch_size=64, data_dir=None, img_size=29, seed: int = 42):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_classes = 10
        max_image_size = 29
        if img_size > max_image_size:
            img_size = max_image_size
        self.img_size = img_size
        self.train_transform = Compose([
            RandomRotation(degrees=(0, 360), interpolation=InterpolationMode.BILINEAR, fill=0),
            Pad((0, 0, 1, 1), fill=0),
            Resize(img_size),
            ToTensor(),
        ])

        self.test_transform = Compose([
            Pad((0, 0, 1, 1), fill=0),
            Resize(img_size),
            ToTensor(),
        ])

        self.generator = torch.Generator().manual_seed(seed)

    def setup(self, stage: str):
        if stage == 'fit':
            self.mnist_full = MnistRotDataset(mode='train', transform=self.train_transform, data_dir=self.data_dir)

            self.mnist_train, self.mnist_val = torch.utils.data.random_split(self.mnist_full, [0.8, 0.2], generator=self.generator)
        if stage == 'test':
            self.mnist_test = MnistRotDataset(mode='test', transform=self.test_transform, data_dir=self.data_dir)
        if stage == 'predict':
            self.mnist_predict = MnistRotDataset(mode='test', transform=self.test_transform, data_dir=self.data_dir)
        
    def train_dataloader(self):
        return TorchDataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=os.cpu_count()//2,  # Use multiple workers
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=True,
            shuffle=True
        )

    def val_dataloader(self):
        return TorchDataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=os.cpu_count()//2,  # Use multiple workers
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=True  # Keep workers alive between epochs
        )

    def test_dataloader(self):
        return TorchDataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=os.cpu_count()//2,  # Use multiple workers
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=True  # Keep workers alive between epochs
        )

    def predict_dataloader(self):
        return TorchDataLoader(
            self.mnist_predict,
            batch_size=self.batch_size,
            num_workers=os.cpu_count()//2,  # Use multiple workers
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=True  # Keep workers alive between epochs
        )


class HFImageTorchDataset(Dataset):
    def __init__(self, hf_split, transform=None, image_key="image", label_key="label"):
        self.ds = hf_split
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex[self.image_key]  # PIL image if decoded by datasets_utils
        if self.transform:
            img = self.transform(img)
        y = int(ex[self.label_key])
        return img, y


# ---------- common transforms (RGB) ----------
def make_transforms(img_size: int) -> Tuple[Compose, Compose]:

    train_tf = Compose([
        Resize(img_size),
        RandomRotation(degrees=(0, 360), interpolation=InterpolationMode.BILINEAR, fill=0),
        RandomVerticalFlip(),
        RandomHorizontalFlip(),
        ToTensor()
    ])

    eval_tf = Compose([
        Resize(img_size),
        ToTensor(),
    ])

    return train_tf, eval_tf

class Resisc45DataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 256, img_size: int = 150, seed: int = 42):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = 45
        max_image_size = 256
        if img_size > max_image_size:
            img_size = max_image_size
        self.img_size = img_size
        self.train_tf, self.eval_tf = make_transforms(img_size)

        self.seed = seed

    def setup(self, stage=None):

        ds = load_dataset("timm/resisc45")

        train_split = ds["train"]
        val_split = ds["validation"]
        test_split = ds["test"]

        self.train_ds = HFImageTorchDataset(train_split, self.train_tf, "image", "label")
        self.val_ds = HFImageTorchDataset(val_split, self.eval_tf,  "image", "label")
        self.test_ds = HFImageTorchDataset(test_split, self.eval_tf,  "image", "label")

    def train_dataloader(self):
        return TorchDataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)

    def val_dataloader(self):
        return TorchDataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=8, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return TorchDataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=8, pin_memory=True, persistent_workers=True)

class EuroSATDataModule(L.LightningDataModule):
    def __init__(self,
                 batch_size: int = 64,
                 img_size: int = 64,
                 seed: int = 42,
                 train_fraction: float = 1.0):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = 10
        max_image_size = 64
        if img_size > max_image_size:
            img_size = max_image_size
        self.img_size = img_size
        self.train_tf, self.eval_tf = make_transforms(img_size)

        self.seed = seed
        if not (0 < train_fraction <= 1.0):
            raise ValueError("train_fraction must be in (0, 1].")
        self.train_fraction = train_fraction

    def setup(self, stage=None):

        ds = load_dataset("blanchon/EuroSAT_RGB")

        train_split = ds["train"]
        val_split = ds["validation"]
        test_split = ds["test"]

        if self.train_fraction < 1.0:
            subset_size = max(1, int(len(train_split) * self.train_fraction))
            train_split = train_split.shuffle(seed=self.seed).select(range(subset_size))

        self.train_ds = HFImageTorchDataset(train_split, self.train_tf, "image", "label")
        self.val_ds = HFImageTorchDataset(val_split, self.eval_tf,  "image", "label")
        self.test_ds = HFImageTorchDataset(test_split, self.eval_tf,  "image", "label")

    def train_dataloader(self):
        return TorchDataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)

    def val_dataloader(self):
        return TorchDataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=8, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return TorchDataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=8, pin_memory=True, persistent_workers=True)
    
# ---------- Colorectal Histology: stratified 85/7.5/7.5 from TRAIN(5000) ----------
class ColorectalHistDataModule(L.LightningDataModule):
    def __init__(self,
                 batch_size: int = 64,
                 img_size: int = 150,
                 seed: int = 42,
                 train_fraction: float = 1.0):
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_classes = 8
        max_image_size = 150
        if img_size > max_image_size:
            img_size = max_image_size
        self.img_size = img_size
        self.train_tf, self.eval_tf = make_transforms(img_size)
        if not (0 < train_fraction <= 1.0):
            raise ValueError("train_fraction must be in (0, 1].")
        self.train_fraction = train_fraction

    def setup(self, stage=None):
        ds = load_dataset("dpdl-benchmark/colorectal_histology")
        full = ds["train"]

        # Cast label column to ClassLabel
        self.num_classes = 8
        full = full.cast_column("label", ClassLabel(num_classes=self.num_classes))

        # 85% train / 15% temp
        split1 = full.train_test_split(test_size=0.15, seed=self.seed, stratify_by_column="label")
        train_split = split1["train"]
        temp_split = split1["test"]

        if self.train_fraction < 1.0:
            subset_size = max(1, int(len(train_split) * self.train_fraction))
            train_split = train_split.shuffle(seed=self.seed).select(range(subset_size))

        # split temp into val/test 50/50 => 7.5% each
        split2 = temp_split.train_test_split(test_size=0.5, seed=self.seed, stratify_by_column="label")
        val_split = split2["train"]
        test_split = split2["test"]

        self.train_ds = HFImageTorchDataset(train_split, self.train_tf, "image", "label")
        self.val_ds = HFImageTorchDataset(val_split, self.eval_tf, "image", "label")
        self.test_ds = HFImageTorchDataset(test_split, self.eval_tf, "image", "label")

    def train_dataloader(self):
        return TorchDataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)

    def val_dataloader(self):
        return TorchDataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=8, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return TorchDataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=8, pin_memory=True, persistent_workers=True)


class ModelNet10PointClouds(L.LightningDataModule):
    def __init__(
        self,
        root: str = "./data/modelnet10",
        num_points: int = 2048,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
        shuffle_points: bool = True,
    ):
        super().__init__()
        self.root = root
        self.num_points = num_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.shuffle_points = shuffle_points

        # Point-cloud transforms: normalize to unit sphere and sample N points per mesh
        self.transform = Compose([
            NormalizeScale(),
            SamplePoints(self.num_points, remove_faces=True),
        ])

    def prepare_data(self):
        # This triggers the download once; PyG handles the URL/zip/unpack for ModelNet10.
        ModelNet(self.root, name="10", train=True)
        ModelNet(self.root, name="10", train=False)

    def setup(self, stage=None):
        train_full = ModelNet(self.root, name="10", train=True, transform=self.transform)
        test = ModelNet(self.root, name="10", train=False, transform=self.transform)

        # Split train into train/val
        n_total = len(train_full)
        n_val = int(self.val_split * n_total)
        n_train = n_total - n_val
        self.train_set, self.val_set = random_split(train_full, [n_train, n_val],
                                                    generator=torch.Generator().manual_seed(42))
        self.test_set = test

    def train_dataloader(self):
        return PyGDataLoader(self.train_set, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return PyGDataLoader(self.val_set, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return PyGDataLoader(self.test_set, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)


if __name__ == "__main__":
    dm = ModelNet10PointClouds(num_points=2048, batch_size=16)
    dm.prepare_data()
    dm.setup()

    # Peek at one batch
    batch = next(iter(dm.train_dataloader()))
    # batch is a PyG Batch with fields like .pos (B*N x 3), .y (labels), and .batch (graph ids)
    print("pos shape:", batch.pos.shape)     # (total_points_in_batch, 3)
    print("labels shape:", batch.y.shape)    # (batch_size,)
    print("num graphs:", batch.num_graphs)
