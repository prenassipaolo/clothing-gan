import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

# download dataset
def download_cifar_datasets():
    # create transform function
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # create datasets
    train_data = CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train,
    )
    test_data = CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test,
    )
    return train_data, test_data
