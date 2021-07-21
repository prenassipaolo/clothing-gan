from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

# download dataset and create dataloader
def download_cifar_datasets():
    train_data = CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    validation_data = CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return train_data, validation_data