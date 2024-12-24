from abc import abstractmethod, ABC

import torch
from torch import nn


class SparseModule(ABC, nn.Module):
    def __init__(self, weight_size, device='cpu'):
        super(SparseModule, self).__init__()
        self.weight_indices = torch.empty(2, 0, dtype=torch.long, device=device)
        self.weight_values = nn.Parameter(torch.empty(0, device=device))
        self.weight_size = list(weight_size)
        self.device = device

    def add_edge(self, child, parent):
        new_edge = torch.tensor([[child, parent]], dtype=torch.long, device=self.device).t()
        self.weight_indices = torch.cat([self.weight_indices, new_edge], dim=1)

        new_weight = torch.empty(1, device=self.device)
        nn.init.uniform_(new_weight)
        self.weight_values.data = torch.cat([self.weight_values.data, new_weight])

    def create_sparse_tensor(self):
        return torch.sparse_coo_tensor(self.weight_indices, self.weight_values, self.weight_size, device=self.device)

    @abstractmethod
    def replace(self, child, parent):
        pass

    def replace_many(self, children, parents):
        for c, p in zip(children, parents):
            self.replace(c, p)


class EmbedLinear(SparseModule):
    def __init__(self, weight_size, activation=nn.ReLU(), device='cpu'):
        super(EmbedLinear, self).__init__([0, weight_size], device=device)
        self.child_counter = 0
        self.activation = activation
        self.device = device

    def replace(self, child, parent):
        self.add_edge(self.child_counter, parent)
        self.weight_size[0] += 1
        self.child_counter += 1

    def forward(self, input):
        sparse_embed_weight = self.create_sparse_tensor()
        output = torch.sparse.mm(sparse_embed_weight, input.t()).t()
        return torch.cat([input, self.activation(output)], dim=1)


class ExpandingLinear(SparseModule):
    def __init__(self, weight: torch.sparse_coo_tensor, bias: torch.sparse_coo_tensor, device='cpu'):
        super(ExpandingLinear, self).__init__(weight.size(), device=device)

        weight = weight.coalesce()
        self.weight_indices = weight.indices().to(device)
        self.weight_values = nn.Parameter(weight.values().to(device))

        self.embed_linears = []

        bias = bias.coalesce()
        self.bias_indices = bias.indices().to(device)
        self.bias_values = nn.Parameter(bias.values().to(device))
        self.bias_size = list(bias.size())

        self.current_iteration = -1
        self.device = device

    def replace(self, child, parent):
        if self.current_iteration == -1:
            self.current_iteration = 0

        if len(self.embed_linears) <= self.current_iteration:
            self.embed_linears.append(EmbedLinear(self.weight_size[1], device=self.device))

        matches = (self.weight_indices[0] == child) & (self.weight_indices[1] == parent)

        assert torch.any(matches), "Edge must extist"

        max_parent = self.weight_indices[1].max().item() + 1 # before deleting edge

        self.weight_indices = self.weight_indices[:, ~matches]
        self.weight_values = nn.Parameter(self.weight_values[~matches])

        self.add_edge(child, max_parent)

        self.weight_size[1] += 1
        self.embed_linears[self.current_iteration].replace(child, parent)

    def replace_many(self, children, parents):
        self.current_iteration += (len(children) != 0 and len(parents) != 0)
        super().replace_many(children, parents)

    def forward(self, input):
        for embed_linear in self.embed_linears:
            input = embed_linear(input)

        sparse_weight = self.create_sparse_tensor()
        sparse_bias = torch.sparse_coo_tensor(self.bias_indices, self.bias_values, self.bias_size,
                                              device=self.device).to_dense()

        output = torch.sparse.mm(sparse_weight, input.t()).t()
        output += sparse_bias.unsqueeze(0)

        return output
