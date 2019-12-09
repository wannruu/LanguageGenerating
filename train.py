import torch
import torch.nn as nn
from .models import TCN, save_model
from .utils import SpeechDataset, one_hot
import numpy as np

def make_batch(batch_size, data):
    B = []
    for i in range(batch_size):
        idx = np.random.randint(0, len(data))
        single_data = data[idx]
        B.append(single_data)
    return torch.stack(B, dim=0)

def train(args):
    from os import path
    import torch.utils.tensorboard as tb
    model = TCN()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    loss = torch.nn.CrossEntropyLoss()

    train_data = SpeechDataset('data/train.txt', transform=one_hot, max_len=args.sequence_length)
    valid_data = SpeechDataset('data/valid.txt', transform=one_hot, max_len=args.sequence_length)
    
    model.train()
    for iterations in range(args.iteration_num):
        batch = make_batch(args.batch_size, train_data)
        batch_data = batch[:,:,:-1]
        batch_data = batch_data.to(device)
        batch_label = batch.argmax(dim=1)
        batch_label = batch_label.to(device)

        o = model(batch_data)
        loss_val = loss(o, batch_label)

        if args.log_dir is not None:
            train_logger.add_scalar('train/loss', loss_val, global_step=iterations)
    
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
    
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1)
    parser.add_argument('-it', '--iteration_num', type=int, default=10000)
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('-sl', '--sequence_length', type=int, default=256)

    args = parser.parse_args()
    train(args)
