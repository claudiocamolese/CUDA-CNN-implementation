from torchvision import datasets, transforms
import torch 

DATASETS = {
    "mnist": datasets.MNIST,
    "fashion": datasets.FashionMNIST
}

def get_dataloader(dataset_name, train=True, batch_size=64, num_workers=2, pin_memory=True):
    dataset_cls = DATASETS.get(dataset_name.lower())
    if dataset_cls is None:
        raise ValueError(f"Unknown dataset {dataset_name}")

    dataset = dataset_cls(
        root="./datasets",
        train=train,
        download=True,
        transform=transforms.ToTensor()
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return loader
