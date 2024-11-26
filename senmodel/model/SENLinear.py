import torch
from torch import nn


class SENLinearBase(nn.Module):

    def __init__(self, initial_weight_size):
        super(SENLinearBase, self).__init__()
