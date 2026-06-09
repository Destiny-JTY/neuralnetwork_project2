"""CIFAR-10 data loaders."""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.datasets as datasets


class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return min(self.n_items, len(self.dataset))


def get_cifar_loader(
    root=None,
    batch_size=128,
    train=True,
    shuffle=None,
    num_workers=0,
    n_items=-1,
    download=True,
    pin_memory=False,
):
    if root is None:
        root = Path(__file__).resolve().parent
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    if shuffle is None:
        shuffle = train

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    data_transforms = transforms.Compose(
        [transforms.ToTensor(),
        normalize])

    dataset = datasets.CIFAR10(
        root=str(root),
        train=train,
        download=download,
        transform=data_transforms,
    )
    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return loader

if __name__ == '__main__':
    train_loader = get_cifar_loader()
    for X, y in train_loader:
        print(X[0])
        print(y[0])
        print(X[0].shape)
        img = np.transpose(X[0], [1,2,0])
        plt.imshow(img*0.5 + 0.5)
        plt.savefig('sample.png')
        print(X[0].max())
        print(X[0].min())
        break
