import torch
from torch import nn


class SENLinearBase(nn.Module):

    def __init__(self, initial_weight_size):
        super(SENLinearBase, self).__init__()
        self.weight_indices = torch.empty(2, 0, dtype=torch.long)
        self.weight_values = nn.Parameter(torch.empty(0))
        self.weight_size = torch.tensor([0, initial_weight_size])

    def replace(self, child, parent, iteration=None):
        new_edge = torch.tensor([[child, parent]], dtype=torch.long).t()
        self.weight_indices = torch.cat([self.weight_indices, new_edge], dim=1)
        new_weight = torch.rand(1)  # New random weight for the edge
        self.weight_values.data = torch.cat([self.weight_values.data, new_weight])

        # Update weight size (increment the row count)
        self.weight_size[0] += 1

    def forward(self, input):
        sparse_weight = torch.sparse_coo_tensor(self.weight_indices, self.weight_values, list(self.weight_size))
        output = torch.sparse.mm(sparse_weight, input.t()).t()
        return output
