# %%
import logging
import os
import argparse
import numpy as np
import torch
from cryptoGPT.model import GPTnopos, GPTConfig
from cryptoGPT.trainer import Trainer, TrainerConfig

from war_and_peace_utils import create_logger, CharDataset

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

# make folder to store the model
save_path = "war_and_peace_expt/l{}_c{}".format(num_layer, context_length)
if not os.path.exists(save_path):
    os.makedirs(save_path)
log_path = os.path.join(save_path, "log")

logger = create_logger(log_path=log_path)


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
mconf = GPTConfig(vocab_size, context_length, n_layer=num_layer, n_head=4, n_embd=64)
model = GPTnopos(mconf)

save_original_path = os.path.join(save_path, "original.pth")
# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=2, batch_size=batch_size, learning_rate=6e-4,
                      ckpt_path=save_original_path,
                      lr_decay=True, warmup_tokens=512 * 20,
                      final_tokens=2 * len(train_dataset) * context_length,
                      num_workers=4)

optimizer = model.configure_optimizers(tconf)

trainer = Trainer(model, train_dataset, None, tconf, optimizer)
trainer.train()

#  store the embedding
char_emb = model.tok_emb.weight.data
emb = char_emb.to('cpu').numpy()
emb /= np.linalg.norm(emb, axis=1, keepdims=True)

avg_result = []
for i in range(n_expt):
    # reinit the embedding of the model
    model.tok_emb.weight.data.normal_(mean=0.0, std=0.02)
    # re-initialize a trainer instance and kick off training
    save_new_path = os.path.join(save_path, "new_{}.pth".format(i))
    tconf = TrainerConfig(max_epochs=2, batch_size=batch_size, learning_rate=6e-4,
                          ckpt_path=save_new_path,
                          lr_decay=True, warmup_tokens=512 * 20,
                          final_tokens=2 * len(test_dataset) * context_length,
                          num_workers=4)
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

    optimizer = model.configure_optimizers(tconf, freeze=True)

    trainer = Trainer(model, test_dataset, None, tconf, optimizer)
    trainer.train()
    #  store the embedding
    char_emb_p = model.tok_emb.weight.data
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
