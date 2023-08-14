import torch

def split_indecies(n , val_pct = 0.2):
    perm = torch.randperm(n)
    split = int((val_pct * n))
    train_idx, val_idx = perm[:split], perm[split:]
    return train_idx, val_idx