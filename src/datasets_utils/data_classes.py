import os
import json
from typing import Tuple
from pathlib import Path
import lightning as L
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset, ClassLabel
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, random_split
from torchvision.transforms import Compose, Pad, Resize, RandomRotation, InterpolationMode, ToTensor, RandomVerticalFlip, RandomHorizontalFlip, Normalize 
from torch_geometric.transforms import NormalizeScale, SamplePoints
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data as PyGData
from torch_geometric.nn import knn_graph, radius_graph
from datasets_utils.mnist_download import download_mnist_rotation

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_STATS_DIR = Path(__file__).resolve().parent / "norm_stats"


def compute_mean_std(ds, batch_size: int = 256, num_workers: int | None = None) -> tuple[list[float], list[float]]:
    nw = num_workers if num_workers is not None else max(0, os.cpu_count() // 2)
    loader = TorchDataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available()
    )
    n, mean, sq_mean = 0, None, None
    with torch.no_grad():
        for x, _ in loader:
            b = x.size(0)
            x = x.view(b, x.size(1), -1)
            batch_mean = x.mean(dim=2).sum(dim=0)
            batch_sq_mean = (x ** 2).mean(dim=2).sum(dim=0)
            if mean is None:
                mean = torch.zeros_like(batch_mean)
                sq_mean = torch.zeros_like(batch_sq_mean)
            mean += batch_mean
            sq_mean += batch_sq_mean
            n += b
    if n == 0:
        raise ValueError("Dataset is empty; cannot compute normalization stats.")
    mean = mean / n
    std = (sq_mean / n - mean ** 2).clamp(min=1e-12).sqrt()
    return mean.tolist(), std.tolist()


def load_or_compute_stats(
    name: str,
    ds,
    stats_dir: Path = DEFAULT_STATS_DIR,
    batch_size: int = 256,
    num_workers: int | None = None
) -> tuple[list[float], list[float]]:
    """
    Load cached normalization stats if present; otherwise compute and persist them.
    """
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / f"{name}_norm.json"
    if stats_path.exists():
        try:
            payload = json.loads(stats_path.read_text())
            if isinstance(payload, dict) and "mean" in payload and "std" in payload:
                return payload["mean"], payload["std"]
            if isinstance(payload, list) and len(payload) == 2:
                return payload[0], payload[1]
        except Exception:
            pass  # fall through to recompute if cache is unreadable
    mean, std = compute_mean_std(ds, batch_size=batch_size, num_workers=num_workers)
    stats_path.write_text(json.dumps({"mean": mean, "std": std}))
    return mean, std


class MnistRotDataset(Dataset):

    def __init__(self, mode, transform=None, data_dir=None):
        assert mode in ['train', 'test']

        self.transform = transform
        root = Path(data_dir) if data_dir else (Path(__file__).resolve().parent / "mnist_rotation_new")
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
    def __init__(self, 
                 batch_size=64, 
                 data_dir=None, 
                 img_size=None, 
                 seed: int = 42, 
                 train_fraction: float = 1.0,
                 aug: bool=True,
                 normalize: bool = True):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.stats_dir = Path(data_dir) if data_dir else DEFAULT_STATS_DIR
        self.num_classes = 10
        default_img_size = 29
        max_image_size = 29
        if img_size is None:
            img_size = default_img_size
        if img_size > max_image_size:
            img_size = max_image_size
        self.img_size = img_size
        self.aug = aug
        self.normalize = normalize
        self.train_transform = None
        self.test_transform = None
        self.generator = torch.Generator().manual_seed(seed)
        if not (0 < train_fraction <= 1.0):
            raise ValueError("train_fraction must be in (0, 1].")
        self.train_fraction = train_fraction

    def prepare_data(self):
        """
        Ensure the rotated MNIST files are present; download and extract if missing.
        """
        root = Path(self.data_dir) if self.data_dir else (Path(__file__).resolve().parent / "mnist_rotation_new")
        download_mnist_rotation(root)

    def setup(self, stage: str | None = None):
        if getattr(self, "_setup_done", False):
            return

        base_transform = Compose([
            Pad((0, 0, 1, 1), fill=0),
            Resize(self.img_size),
            ToTensor(),
        ])
        full_train_ds = MnistRotDataset(mode='train', transform=base_transform, data_dir=self.data_dir)

        # Optional sub-sampling of the train set
        indices = torch.randperm(len(full_train_ds), generator=self.generator).tolist()
        if self.train_fraction < 1.0:
            subset_size = max(1, int(len(indices) * self.train_fraction))
            indices = indices[:subset_size]

        n_total = len(indices)
        n_train = int(0.8 * n_total)
        n_val = n_total - n_train
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_subset = torch.utils.data.Subset(full_train_ds, train_indices)

        mean, std = None, None
        if self.normalize:
            mean, std = load_or_compute_stats(
                "mnist_rot",
                train_subset,
                stats_dir=self.stats_dir,
                num_workers=os.cpu_count() // 2
            )

        train_ops = []
        if self.aug:
            train_ops.append(RandomRotation(degrees=(0, 360), interpolation=InterpolationMode.BILINEAR, fill=0))
        train_ops.extend([
            Pad((0, 0, 1, 1), fill=0),
            Resize(self.img_size),
            ToTensor(),
        ])
        eval_ops = [
            Pad((0, 0, 1, 1), fill=0),
            Resize(self.img_size),
            ToTensor(),
        ]
        if mean is not None and std is not None:
            norm = Normalize(mean, std)
            train_ops.append(norm)
            eval_ops.append(norm)

        self.train_transform = Compose(train_ops)
        self.test_transform = Compose(eval_ops)

        train_base = MnistRotDataset(mode='train', transform=self.train_transform, data_dir=self.data_dir)
        eval_base = MnistRotDataset(mode='train', transform=self.test_transform, data_dir=self.data_dir)

        self.mnist_train = torch.utils.data.Subset(train_base, train_indices)
        self.mnist_val = torch.utils.data.Subset(eval_base, val_indices)
        self.mnist_test = MnistRotDataset(mode='test', transform=self.test_transform, data_dir=self.data_dir)
        self.mnist_predict = MnistRotDataset(mode='test', transform=self.test_transform, data_dir=self.data_dir)
        self._setup_done = True
        
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
def make_transforms(
    img_size: int,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    aug: bool = True
) -> Tuple[Compose, Compose]:

    train_ops = [Resize(img_size)]
    if aug:
        train_ops.extend([
            RandomRotation(degrees=(0, 360), interpolation=InterpolationMode.BILINEAR, fill=0),
            RandomVerticalFlip(),
            RandomHorizontalFlip(),
        ])
    train_ops.append(ToTensor())

    eval_ops = [
        Resize(img_size),
        ToTensor(),
    ]

    if mean is not None and std is not None:
        norm = Normalize(mean, std)
        train_ops.append(norm)
        eval_ops.append(norm)

    return Compose(train_ops), Compose(eval_ops)

class Resisc45DataModule(L.LightningDataModule):
    def __init__(self, 
                 batch_size: int = 256, 
                 img_size: int | None = None, 
                 seed: int = 42, 
                 train_fraction: float = 1.0, 
                 aug: bool=True,
                 normalize: bool = True):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = 45
        default_img_size = 256
        max_image_size = 256
        if img_size is None:
            img_size = default_img_size
        if img_size > max_image_size:
            img_size = max_image_size
        self.img_size = img_size
        self.aug = aug
        self.normalize = normalize
        self.train_tf = None
        self.eval_tf = None
        self.stats_dir = DEFAULT_STATS_DIR
        self.seed = seed
        if not (0 < train_fraction <= 1.0):
            raise ValueError("train_fraction must be in (0, 1].")
        self.train_fraction = train_fraction

    def setup(self, stage=None):

        ds = load_dataset("timm/resisc45")

        train_split = ds["train"]
        val_split = ds["validation"]
        test_split = ds["test"]

        if self.train_fraction < 1.0:
            subset_size = max(1, int(len(train_split) * self.train_fraction))
            train_split = train_split.shuffle(seed=self.seed).select(range(subset_size))

        base_tf = Compose([Resize(self.img_size), ToTensor()])
        base_train_ds = HFImageTorchDataset(train_split, base_tf, "image", "label")
        mean, std = None, None
        if self.normalize:
            mean, std = load_or_compute_stats(
                "resisc45",
                base_train_ds,
                stats_dir=self.stats_dir,
                num_workers=8
            )

        self.train_tf, self.eval_tf = make_transforms(self.img_size, mean, std, aug=self.aug)

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
                 img_size: int | None = None,
                 seed: int = 42,
                 train_fraction: float = 1.0,
                 aug: bool=True,
                 normalize: bool = True):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = 10
        default_img_size = 64
        max_image_size = 64
        if img_size is None:
            img_size = default_img_size
        if img_size > max_image_size:
            img_size = max_image_size
        self.img_size = img_size
        self.aug = aug
        self.normalize = normalize
        self.train_tf = None
        self.eval_tf = None
        self.stats_dir = DEFAULT_STATS_DIR
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

        base_tf = Compose([Resize(self.img_size), ToTensor()])
        base_train_ds = HFImageTorchDataset(train_split, base_tf, "image", "label")
        mean, std = None, None
        if self.normalize:
            mean, std = load_or_compute_stats(
                "eurosat",
                base_train_ds,
                stats_dir=self.stats_dir,
                num_workers=8
            )

        self.train_tf, self.eval_tf = make_transforms(self.img_size, mean, std, aug=self.aug)

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
                 img_size: int | None = None,
                 seed: int = 42,
                 train_fraction: float = 1.0,
                 aug: bool =True,
                 normalize: bool = True):
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_classes = 8
        default_img_size = 150
        max_image_size = 150
        if img_size is None:
            img_size = default_img_size
        if img_size > max_image_size:
            img_size = max_image_size
        self.img_size = img_size
        self.aug = aug
        self.normalize = normalize
        self.train_tf = None
        self.eval_tf = None
        self.stats_dir = DEFAULT_STATS_DIR
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

        base_tf = Compose([Resize(self.img_size), ToTensor()])
        base_train_ds = HFImageTorchDataset(train_split, base_tf, "image", "label")
        mean, std = None, None
        if self.normalize:
            mean, std = load_or_compute_stats(
                "colorectal_hist",
                base_train_ds,
                stats_dir=self.stats_dir,
                num_workers=8
            )

        self.train_tf, self.eval_tf = make_transforms(self.img_size, mean, std, aug=self.aug)

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


class MnistRotPointCloudDataset(Dataset):
    """
    PyTorch Dataset for MNIST rotation point clouds.
    Loads pre-converted point cloud data from .npz files.
    """

    def __init__(self,
                 npz_path: Path,
                 transform=None,
                 augment_rotation: bool = False,
                 k_neighbors: int = 8,
                 radius: float = None):
        """
        Args:
            npz_path: Path to .npz file containing point clouds
            transform: Optional transform to apply to features
            augment_rotation: Whether to apply random rotation augmentation
            k_neighbors: Number of neighbors for kNN graph construction
            radius: If provided, use radius graph instead of kNN (None disables)
        """
        self.npz_path = npz_path
        self.transform = transform
        self.augment_rotation = augment_rotation
        self.k_neighbors = k_neighbors
        self.radius = radius

        # Load data
        data = np.load(npz_path, allow_pickle=True)
        self.coords = data['coords']  # Object array of variable-length arrays
        self.features = data['features']  # Object array of variable-length arrays
        self.labels = data['labels']  # (N,) int64 array
        self.num_points = data['num_points']  # (N,) int array

    def __len__(self):
        return len(self.labels)

    def _rotate_pointcloud_2d(self, coords: np.ndarray, angle: float = None) -> np.ndarray:
        """
        Rotate 2D point cloud by a random or specified angle.

        Args:
            coords: (N, 2) array of (x, y) coordinates
            angle: Rotation angle in radians. If None, samples uniformly from [0, 2π)

        Returns:
            Rotated coordinates (N, 2)
        """
        if angle is None:
            angle = np.random.uniform(0, 2 * np.pi)

        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ], dtype=np.float32)

        return coords @ rotation_matrix.T

    def __getitem__(self, idx):
        """
        Returns a PyG Data object with:
            - x: features (num_points, feature_dim)
            - pos: coordinates (num_points, 2)
            - edge_index: graph connectivity (2, num_edges)
            - y: label (scalar)
        """
        coords = self.coords[idx].copy()  # (N, 2)
        features = self.features[idx].copy()  # (N, 1)
        label = int(self.labels[idx])

        # Apply rotation augmentation
        if self.augment_rotation:
            coords = self._rotate_pointcloud_2d(coords)

        # Convert to tensors
        pos = torch.from_numpy(coords).float()  # (N, 2)
        x = torch.from_numpy(features).float()  # (N, 1)

        # Build graph connectivity
        if self.radius is not None:
            # Radius graph
            edge_index = radius_graph(pos, r=self.radius, loop=False)
        else:
            # kNN graph
            edge_index = knn_graph(pos, k=self.k_neighbors, loop=False)

        # Create PyG Data object
        data = PyGData(x=x, pos=pos, edge_index=edge_index, y=label)

        return data


class MnistRotPointCloudDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for MNIST rotation point clouds.

    Expected directory structure:
        <data_dir>/
            train.npz
            test.npz
            metadata.txt

    To generate point cloud data, first run:
        python src/datasets_utils/mnist_to_pointcloud.py
    """

    def __init__(self,
                 batch_size: int = 64,
                 data_dir: Path | str = None,
                 seed: int = 42,
                 train_fraction: float = 1.0,
                 augment_rotation: bool = True,
                 k_neighbors: int = 8,
                 radius: float = None,
                 num_workers: int = None):
        """
        Args:
            batch_size: Batch size for dataloaders
            data_dir: Directory containing point cloud .npz files
            seed: Random seed for train/val split
            train_fraction: Fraction of training data to use (for experiments)
            augment_rotation: Whether to apply random rotation augmentation during training
            k_neighbors: Number of neighbors for kNN graph construction
            radius: If provided, use radius graph instead of kNN
            num_workers: Number of dataloader workers (defaults to cpu_count//2)
        """
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.train_fraction = train_fraction
        self.augment_rotation = augment_rotation
        self.k_neighbors = k_neighbors
        self.radius = radius
        self.num_workers = num_workers if num_workers is not None else max(1, os.cpu_count() // 2)
        self.num_classes = 10

        # Set data directory
        if data_dir is None:
            self.data_dir = Path(__file__).resolve().parent / "mnist_rotation_pointcloud"
        else:
            self.data_dir = Path(data_dir)

        if not (0 < train_fraction <= 1.0):
            raise ValueError("train_fraction must be in (0, 1].")

        self.generator = torch.Generator().manual_seed(seed)

    def prepare_data(self):
        """
        Check if point cloud files exist. If not, provide instructions.
        """
        train_file = self.data_dir / "train.npz"
        test_file = self.data_dir / "test.npz"

        if not train_file.exists() or not test_file.exists():
            raise FileNotFoundError(
                f"Point cloud files not found in {self.data_dir}.\n"
                f"Please run the conversion script first:\n"
                f"  python src/datasets_utils/mnist_to_pointcloud.py --output_dir {self.data_dir}"
            )

        print(f"Found point cloud data in {self.data_dir}")

    def setup(self, stage: str | None = None):
        if getattr(self, "_setup_done", False):
            return

        train_npz = self.data_dir / "train.npz"
        test_npz = self.data_dir / "test.npz"

        # Create full training dataset
        full_train_ds = MnistRotPointCloudDataset(
            train_npz,
            augment_rotation=False,  # No augmentation for computing stats
            k_neighbors=self.k_neighbors,
            radius=self.radius
        )

        # Split into train/val
        indices = torch.randperm(len(full_train_ds), generator=self.generator).tolist()

        if self.train_fraction < 1.0:
            subset_size = max(1, int(len(indices) * self.train_fraction))
            indices = indices[:subset_size]

        n_total = len(indices)
        n_train = int(0.8 * n_total)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        # Create train dataset with augmentation
        train_full = MnistRotPointCloudDataset(
            train_npz,
            augment_rotation=self.augment_rotation,
            k_neighbors=self.k_neighbors,
            radius=self.radius
        )

        # Create eval datasets without augmentation
        eval_full = MnistRotPointCloudDataset(
            train_npz,
            augment_rotation=False,
            k_neighbors=self.k_neighbors,
            radius=self.radius
        )

        self.mnist_train = torch.utils.data.Subset(train_full, train_indices)
        self.mnist_val = torch.utils.data.Subset(eval_full, val_indices)

        self.mnist_test = MnistRotPointCloudDataset(
            test_npz,
            augment_rotation=False,
            k_neighbors=self.k_neighbors,
            radius=self.radius
        )

        self.mnist_predict = self.mnist_test

        print(f"Point cloud dataset setup complete:")
        print(f"  - Train samples: {len(self.mnist_train)}")
        print(f"  - Val samples: {len(self.mnist_val)}")
        print(f"  - Test samples: {len(self.mnist_test)}")
        print(f"  - Graph construction: {'radius' if self.radius else 'kNN'}")
        if self.radius:
            print(f"    Radius: {self.radius}")
        else:
            print(f"    k-neighbors: {self.k_neighbors}")

        self._setup_done = True

    def train_dataloader(self):
        return PyGDataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return PyGDataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        return PyGDataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def predict_dataloader(self):
        return PyGDataLoader(
            self.mnist_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )


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
