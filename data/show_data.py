import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from download_data import download_cifar_dataset, download_mnist_dataset

BATCH_SIZE = 4
NUM_WORKERS = 2

cifar_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
mnist_classes = range(0,10)

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    return 



if __name__ == '__main__':

    # load data
    #trainset, testset = download_cifar_dataset()
    #classes = cifar_classes
    trainset, testset = download_mnist_dataset()
    classes = mnist_classes

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

