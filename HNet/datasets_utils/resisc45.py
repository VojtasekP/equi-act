from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms
from datasets import load_dataset


class HFImageTorchDataset(torch.utils.data.Dataset):
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

class Resisc45DataModule(L.LightningDataModule):
    def __init__(self, batch_size=128, num_workers=4, img_size=28, to_grayscale=True):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        t = []
        if to_grayscale:
            t.append(transforms.Grayscale())
        t += [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        self.transform = transforms.Compose(t)
        self.num_classes = 45  # RESISC45 has 45 classes

    def prepare_data(self):
        load_dataset("timm/resisc45")

    def setup(self, stage=None):
        ds_all = load_dataset("timm/resisc45")
        # Try to use provided splits; if no val split, create one
        if "validation" in ds_all:
            train_split = ds_all["train"]
            val_split   = ds_all["validation"]
            test_split  = ds_all["test"] if "test" in ds_all else ds_all["validation"]
        else:
            # common HF version has only train; make val/test from train
            tmp = ds_all["train"].train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
            train_split = tmp["train"]
            tmp2 = tmp["test"].train_test_split(test_size=0.5, seed=42, stratify_by_column="label")
            val_split   = tmp2["train"]
            test_split  = tmp2["test"]

        self.train_ds = HFImageTorchDataset(train_split, self.transform)
        self.val_ds   = HFImageTorchDataset(val_split,   self.transform)
        self.test_ds  = HFImageTorchDataset(test_split,  self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)
