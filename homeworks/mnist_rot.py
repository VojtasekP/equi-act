import lightning as L
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Pad, Resize, RandomRotation, InterpolationMode, ToTensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MnistRotDataset(Dataset):

    def __init__(self, mode, transform=None):
        assert mode in ['train', 'test']

        if mode == "train":
            file = "mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
        else:
            file = "mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"

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


class MnistRotDataModule(L.LightningModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

        self.train_transform = Compose([
            Pad((0, 0, 1, 1), fill=0),  # Pad the image
            Resize(87),  # Upscale
            RandomRotation(180.0, interpolation=InterpolationMode.BILINEAR, expand=False),  # Random rotate
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
    def setup(self, stage:str):
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
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)