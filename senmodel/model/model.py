import torch
import torch.nn as nn



class EmbedLinear(nn.Module):
    def __init__(self, weight_size):
        super(EmbedLinear, self).__init__()
        self.weight_indices = torch.empty(2, 0, dtype=torch.int)
        self.weight_values = nn.Parameter(torch.empty(0))
        self.weight_size = torch.tensor([0, weight_size])
        self.child_counter = 0

    def replace(self, child, parent, iteration=None):
        new_edge = torch.tensor([[self.child_counter, parent]], dtype=torch.long).t()
        self.weight_indices = torch.cat([self.weight_indices, new_edge], dim=1)
        self.weight_values.data = torch.cat([self.weight_values, torch.rand(1)])  # todo: fix torch.rand(1)
        self.weight_size[0] += 1
        self.child_counter += 1

    def forward(self, input):
        sparse_embed_weight = torch.sparse_coo_tensor(self.weight_indices, self.weight_values, list(self.weight_size))

        output = torch.sparse.mm(sparse_embed_weight, input.t()).t()
        return torch.cat([input, output], dim=1)


class ExpandingLinear(nn.Module):
    def __init__(self, weight: torch.sparse_coo_tensor, bias: torch.sparse_coo_tensor):
        super(ExpandingLinear, self).__init__()

        # sparse weight
        self.weight_indices = weight.coalesce().indices()
        self.weight_values = nn.Parameter(weight.coalesce().values())
        self.weight_size = list(weight.coalesce().size())

        self.embed_linears = []

        # TODO think about bias representation
        self.bias_indices = bias.coalesce().indices()
        self.bias_values = nn.Parameter(bias.coalesce().values())
        self.bias_size = list(bias.coalesce().size())

        self.last_iteration = -1

    def replace(self, child, parent, iteration):
        if iteration > self.last_iteration:
            self.last_iteration = iteration
            self.embed_linears.append(EmbedLinear(self.weight_size[1]))
        matches = (self.weight_indices[0] == child) & (self.weight_indices[1] == parent)

        self.weight_indices = self.weight_indices[:, ~matches]
        self.weight_values = nn.Parameter(self.weight_values[~matches])

        max_parent = self.weight_indices[1].max().item() + 1
        new_edge = torch.tensor([[child, max_parent]], dtype=torch.long).t()
        self.weight_indices = torch.cat([self.weight_indices, new_edge], dim=1)

        self.weight_values.data = torch.cat([self.weight_values.data, torch.rand(1)])

        self.weight_size[1] += 1
        self.embed_linears[iteration].replace(child, parent)

    def forward(self, input):
        for i in range(self.last_iteration + 1):
            input = self.embed_linears[i](input)

        sparse_weight = torch.sparse_coo_tensor(self.weight_indices, self.weight_values, self.weight_size)
        sparse_bias = torch.sparse_coo_tensor(self.bias_indices, self.bias_values, self.bias_size).to_dense()

        output = torch.sparse.mm(sparse_weight, input.t()).t()

        output += sparse_bias.unsqueeze(0)

        return output
