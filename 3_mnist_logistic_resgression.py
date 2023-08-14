import torch 
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from utils import *

import matplotlib.pyplot as plt


class MnistModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(28*28, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        out = self.linear(x)
        out = torch.nn.functional.log_softmax(out, dim=1)
        return out
    
    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.forward(x), y)
    
    def accuracy(self, x, y):
        return (torch.argmax(self.forward(x), dim=1) == y).float().mean()
    


if __name__ == '__main__':
    # download the MNIST dataset with pytorch
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

    # show an image
    plt.imshow(mnist.data[0].numpy().reshape(28, 28), cmap='gray')
    plt.show()

    # split the dataset into train and validation
    train_idx, val_idx = split_indecies(len(mnist))
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(mnist, batch_size= 100, sampler=train_sampler)
    val_loader = DataLoader(mnist, batch_size= 100, sampler=val_sampler)


    model = MnistModel()

    x =1



