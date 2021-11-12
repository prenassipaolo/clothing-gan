import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from data.download_data import download_cifar_dataset, download_mnist_dataset
from data.show_data import imshow
#from data import download_data as dd
#from data import show_data as sd


BATCH_SIZE = 4
NUM_WORKERS = 2

CIFAR_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
MNIST_CLASSES = range(0,10)

trainset, testset = download_mnist_dataset()
classes = MNIST_CLASSES

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                        shuffle=True, num_workers=NUM_WORKERS)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                        shuffle=False, num_workers=NUM_WORKERS)


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(BATCH_SIZE)))