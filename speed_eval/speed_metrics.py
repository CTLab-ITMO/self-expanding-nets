import numpy as np
import torch
from torch import nn
import time

from train import train_one_model, valid_model


def get_number_of_params(model: nn.Module) -> int:
    n_params = 0
    for name, param in model.named_parameters():
        if param.is_sparse:
            n_params += param._nnz()
        else:
            n_params += param.numel()
    return n_params


def get_train_time(model, train_loader, valid_loader, criterion, epochs):
    start_time = time.time()
    train_one_model(model, train_loader, valid_loader, criterion=criterion, epochs=epochs)
    return time.time() - start_time


def get_valid_time(model, valid_loader, criterion, device):
    start_time = time.time()
    valid_model(model, valid_loader, criterion, device)
    return time.time() - start_time


def get_avg_inference_time(model, dataloader, device):
    measures = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            start_time = time.time()
            x, y = x.to(device), y.to(device)
            output = model(x).view(-1)
            measures.append(time.time() - start_time)
    return np.mean(measures)


def inference_time_over_params(model: nn.Module, 
                               dataloader: torch.utils.data.DataLoader, 
                               device: torch.device) -> float:
    n_params = get_number_of_params(model)
    inference_time = get_avg_inference_time(model, dataloader, device)
    return inference_time * 1000 / n_params
