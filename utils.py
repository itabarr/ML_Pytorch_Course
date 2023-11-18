import torch    

def split_indecies(n , val_pct = 0.2):
    perm = torch.randperm(n)
    split = int((val_pct * n))
    train_idx, val_idx = perm[:split], perm[split:]
    return train_idx, val_idx

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    history = [] # for recording epoch-wise results
    best_val_acc = 0

    for epoch in range(epochs):
        
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)

        if result['val_acc'] > best_val_acc:
            best_val_acc = result['val_acc'] 
            torch.save(model.state_dict(), f'best_model.pth')
            print("Saved best model.")

        history.append(result)

    return history

def accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    good_preds = (preds == labels).float()
    return torch.mean(good_preds)

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')

    return torch.device('cpu')

def to_device(data , device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)



class DeviceDataLoader():
    def __init__(self , dl , device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        for d in self.dl:
            yield to_device(d, self.device)

    def __len__(self):
        return len(self.dl)
