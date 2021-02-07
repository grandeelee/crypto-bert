import logging
from torch.utils.data import Dataset
import torch


def create_logger(log_path=None):
    # create log formatter
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # create root logger and set level to debug
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if log_path:
        # create file handler and set level to debug
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # logger.propagate = False

    return logger


class CharDataset(Dataset):

    def __init__(self, data, block_size):
        data_size = len(data)
        print('data has %d characters' % (data_size))
        self.block_size = block_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        dix = self.data[idx:idx + self.block_size + 1]

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
