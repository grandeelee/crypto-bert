"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 4 # for DataLoader
    shuffle = False
    finetune = False
    mask = False

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_en_dataset, test_zh_dataset, test_cs_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_en_dataset = test_en_dataset
        self.test_zh_dataset = test_zh_dataset
        self.test_cs_dataset = test_cs_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, suffix):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path + suffix)
        torch.save(raw_model.state_dict(), self.config.ckpt_path + '.' + suffix)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        self.best_en_loss = float('inf')
        self.best_zh_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        
        def run_epoch(split):
            is_train = split == 'train'
            is_en = split == 'test_en'
            is_zh = split == 'test_zh'
            is_cs = split == 'test_cs'
            model.train(is_train)
            if is_train:
                data = self.train_dataset
            elif is_en:
                data = self.test_en_dataset
            elif is_zh:
                data = self.test_zh_dataset
            elif is_cs:
                data = self.test_cs_dataset
            else:
                data = None
            loader = DataLoader(data, shuffle=config.shuffle, pin_memory=False,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, data in pbar:
                x, y = data
                pos, mask = None, None
                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, targets=y, pos=pos, mask=mask)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())


            if is_en:
                test_loss = float(np.mean(losses))
                logger.info("test en loss: %f", test_loss)
                return test_loss
            if is_zh:
                test_loss = float(np.mean(losses))
                logger.info("test zh loss: %f", test_loss)
                return test_loss
            if is_cs:
                test_loss = float(np.mean(losses))
                logger.info("test cs loss: %f", test_loss)
                return test_loss

        if self.test_en_dataset is not None:
            test_en_loss = run_epoch('test_en')
        if self.test_zh_dataset is not None:
            test_zh_loss = run_epoch('test_zh')
        if self.test_cs_dataset is not None:
            test_cs_loss = run_epoch('test_cs')