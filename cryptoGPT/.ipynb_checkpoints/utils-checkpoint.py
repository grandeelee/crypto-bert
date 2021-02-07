import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x

class train_dataset(Dataset):

    def __init__(self, data, block_size):
        data_size = len(data)
        print('data has %d words.' % (data_size))
        self.block_size = block_size
        self.data = data
    
    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx * self.block_size:(idx + 1) * self.block_size + 1]

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
    
class my_dataset(Dataset):

    def __init__(self, data, block_size):
        data_size = len(data)
        print('data has %d words.' % (data_size))
        self.block_size = block_size
        self.data = data
    
    def __len__(self):
        return len(self.data) // (self.block_size // 2) - 1

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        start = idx * (self.block_size // 2)
        end = idx * (self.block_size // 2) + (self.block_size)
        chunk = self.data[start:end+1]

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
    
class line_by_line_dataset(Dataset):

    def __init__(self, data, block_size=128):
        assert block_size == 128, "block size preset to be 128"
        data_size = len(data)
        print('data has %d lines.' % (data_size))
        self.block_size = block_size
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx]

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
    
class para_finetune_dataset(Dataset):

    def __init__(self, data, block_size=128):
        assert block_size == 128, "block size preset to be 128"
        data_size = len(data)
        print('data has %d lines.' % (data_size))
        self.block_size = block_size
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx]['sent']
        pos = torch.tensor(self.data[idx]['pos'], dtype=torch.long)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y, pos

    
class mask_finetune_dataset(Dataset):

    def __init__(self, data, block_size=128):
        assert block_size == 128, "block size preset to be 128"
        data_size = len(data)
        print('data has %d lines.' % (data_size))
        self.block_size = block_size
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx]['sent']
        pos = torch.tensor(self.data[idx]['pos'], dtype=torch.long)
        mask = torch.tensor(self.data[idx]['mask'], dtype=torch.bool)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y, pos, mask
    