# %%
import logging
import os
import argparse
from tqdm import tqdm
import math
import numpy as np
import torch
from cryptoGPT.model import RNNModel
from cryptoGPT.trainer import TrainerConfig
from torch.utils.data.dataloader import DataLoader
from war_and_peace_utils import create_logger, CharDataset

# %%
parser = argparse.ArgumentParser()
parser.add_argument("context_length", type=int, default=5)
parser.add_argument("n_layer", type=int, default=12)
parser.add_argument("batch_size", type=int, default=64)
parser.add_argument("n_expt", type=int, default=1)
arg = parser.parse_args()

context_length = arg.context_length
num_layer = arg.n_layer
batch_size = arg.batch_size
n_expt = arg.n_expt

if not os.path.exists("war_and_peace_expt_lstm"):
    os.makedirs("war_and_peace_expt_lstm")
save_path = "war_and_peace_expt_lstm/l{}_c{}".format(num_layer, context_length)
if not os.path.exists(save_path):
    os.makedirs(save_path)
log_path = os.path.join(save_path, "log")

logger = create_logger(log_path=log_path)
# %%
with open('war_and_peace.txt', 'r') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
word2idx = {j: i for i, j in enumerate(chars)}
idx2word = {i: j for i, j in enumerate(chars)}
pkey = np.random.permutation(vocab_size)
ipkey = np.argsort(pkey)
indexed_text = [word2idx[i] for i in text]
permuted_text = [pkey[i] for i in indexed_text]

train_dataset = CharDataset(indexed_text, context_length)
test_dataset = CharDataset(permuted_text, context_length)

# init a new model
model = RNNModel('LSTM', vocab_size, 64, 64, num_layer)
params = [p for pn, p in model.named_parameters() if p.requires_grad]
save_original_path = os.path.join(save_path, "original.pth")
tconf = TrainerConfig(max_epochs=2, batch_size=batch_size, learning_rate=6e-4,
                      ckpt_path=save_original_path,
                      lr_decay=True, warmup_tokens=512 * 20,
                      final_tokens=2 * len(train_dataset) * context_length,
                      num_workers=4)
optimizer = torch.optim.Adam(params, lr=tconf.learning_rate, betas=tconf.betas)
criterion = torch.nn.NLLLoss()
device = 'cpu'
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    model = torch.nn.DataParallel(model).to(device)


# %%

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# Turn on training mode which enables dropout.

def train(model, optimizer):
    model.train()
    hidden = model.init_hidden(batch_size)
    loader = DataLoader(train_dataset, shuffle=True, pin_memory=True,
                        batch_size=batch_size,
                        num_workers=4)
    pbar = tqdm(enumerate(loader), total=len(loader))

    tokens = 0
    for epoch in range(tconf.max_epochs):
        for it, (x, y) in pbar:
            # place data on the correct device
            x = x.to(device).permute(1, 0)
            y = y.to(device).permute(1, 0).reshape(-1)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            model.zero_grad()
            hidden = repackage_hidden(hidden)
            output, hidden = model(x, hidden)
            loss = criterion(output, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
            if tokens < tconf.warmup_tokens:
                # linear warmup
                lr_mult = float(tokens) / float(max(1, tconf.warmup_tokens))
            else:
                # cosine learning rate decay
                progress = float(tokens - tconf.warmup_tokens) / float(
                    max(1, tconf.final_tokens - tconf.warmup_tokens))
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            lr = tconf.learning_rate * lr_mult
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")


train(model, optimizer)
# %%
char_emb = model.encoder.weight.data
emb = char_emb.to('cpu').numpy()
emb /= np.linalg.norm(emb, axis=1, keepdims=True)

# %%
avg_result = []
for i in range(n_expt):
    # reinit the embedding of the model
    model.encoder.weight.data.uniform_(-0.1, 0.1)
    # re-initialize a trainer instance and kick off training
    save_new_path = os.path.join(save_path, "new_{}.pth".format(i))
    tconf = TrainerConfig(max_epochs=2, batch_size=batch_size, learning_rate=6e-4,
                          ckpt_path=save_new_path,
                          lr_decay=True, warmup_tokens=512 * 20,
                          final_tokens=2 * len(test_dataset) * context_length,
                          num_workers=4)
    for pn, p in model.named_parameters():
        if "encoder" not in pn:
            p.requires_grad = False
    params = [p for pn, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=tconf.learning_rate, betas=tconf.betas)
    train(model, optimizer)
    #  store the embedding
    char_emb_p = model.encoder.weight.data
    emb_p = char_emb_p.to('cpu').numpy()
    # normalize embedding for self similarity
    emb_p /= np.linalg.norm(emb_p, axis=1, keepdims=True)
    # get the restored key
    ssm = emb_p @ emb.T
    restored_key = np.argmax(ssm, axis=1)
    acc = sum(ipkey == restored_key) / len(ipkey) * 100

    avg_result.append(acc)
    logging.info("for layer: {}, context length: {}, run: {}, acc :{}".format(num_layer, context_length, i, acc))

logging.info("for layer: {}, context length: {}, ave acc :{}".format(num_layer, context_length,
                                                                     sum(avg_result) / len(avg_result)))
