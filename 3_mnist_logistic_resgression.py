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
        out = torch.nn.functional.softmax(out, dim=1)
        return out
    
    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.forward(x), y)
    

def accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    good_preds = (preds == labels).float()
    return torch.mean(good_preds)


    


if __name__ == '__main__':
    # download the MNIST dataset with pytorch
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

    # split the dataset into train and validation
    train_idx, val_idx = split_indecies(len(mnist))
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(mnist, batch_size= 100, sampler=train_sampler)
    val_loader = DataLoader(mnist, batch_size= 100, sampler=val_sampler)

    model = MnistModel()

    #print one batch information
    for images, labels in train_loader:
        print(f"This is the first batch of size: {images.shape}")
        print(f"First 10 labels: {labels.data[:10]}")
        
        outputs = model(images)
        max_idx = torch.argmax(outputs, dim=1)
        print(f"First 10 models result: {max_idx.data[:10]}")

        # averaging each number , initialy should be around 0.1 for each number
        average_prob = torch.mean(outputs, dim=0)
        print(f"The average probability of each number is: {average_prob[:10]}")

        acc = accuracy(outputs, labels)
        # print the accuracy of the model with 2 digits precision
        print(f"The accuracy of the model is: {acc:.2f}") 

        plt.imshow(images[0].numpy().reshape(28, 28), cmap='gray')
        plt.show()
        
        break







