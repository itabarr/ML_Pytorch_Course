import torch 
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils import *

import matplotlib.pyplot as plt



class MnistModel(torch.torch.nn.Module):
    def __init__(self , in_size , hidden_size , out_size):
        super().__init__()
        self.best_acc = 1
        self.linear_1 = torch.nn.Linear(in_size, hidden_size)
        self.linear_2 = torch.nn.Linear(hidden_size, out_size)
        
    def forward(self, xb):
        xb = xb.view(xb.size(0) , -1)

        out = self.linear_1(xb)
        out = F.relu(out)
        out = self.linear_2(out)
         
        return out
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        

        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
    

if __name__ == '__main__':

    #Evaluate the model on few samples
    model = MnistModel(in_size= 28 * 28 , hidden_size= 32 , out_size= 10)
    model.load_state_dict(torch.load("best_model.pth"))
    


    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

    # split the dataset into train and validation
    train_idx, val_idx = split_indecies(len(mnist))
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(mnist, batch_size= 100, sampler=train_sampler)
    val_loader = DataLoader(mnist, batch_size= 100, sampler=val_sampler)

    # move images to gpu
    device = get_default_device()

    i = 0
    for images , labels in val_loader:
        image = images[0]
        
        result = model(image)
        probs = F.softmax(result , dim=-1)
        
        top_probs, top_indices = torch.topk(probs, 3)
        top_probs = top_probs.squeeze()
        top_indices = top_indices.squeeze()

        plt.imshow(torch.squeeze(image), cmap='gray')
        for i in range(top_probs.size(0)):
            print(f"Number {top_indices[i].item()} with probability {top_probs[i].item():.4f}")
        plt.show()

        if i > 9:
            break
        
        i =+ 1

    plt.show()
    # model = MnistModel(in_size= 28 * 28 , hidden_size= 32 , out_size= 10)

    # # Move model and data to GPU
    # to_device(model, device)
    # train_loader = DeviceDataLoader(train_loader, device)
    # val_loader = DeviceDataLoader(val_loader, device)        

    # fit(100 , 0.01 , model , train_loader , val_loader)






