import torch
import torch.nn as nn


class EmbedLinear(nn.Module):
    def __init__(self, weight_size):
        super(EmbedLinear, self).__init__()
        self.weight_indeces = torch.empty(2, 0, dtype=torch.int)
        self.weight_values = nn.Parameter(torch.empty(0))
        self.weight_size = torch.tensor([0, weight_size])

        self.child_counter = 0

    def replace(self, child, parent):
        # add new edge to embed weight
        # where self.child_counter is number of nodes in embed weight
        # and parent is number of input node
        self.weight_indeces = torch.cat([self.weight_indeces, torch.tensor([[self.child_counter, parent]]).t()], dim=1)
        # todo smart weight generation
        self.weight_values.data = torch.cat(
            [self.weight_values, torch.rand(1)])  # add new random weight to end of embed value list
        self.weight_size[0] += 1
        self.child_counter += 1

    def forward(self, input):
        # create real sparse weight
        sparse_embed_weight = torch.sparse.FloatTensor(
            self.weight_indeces,
            self.weight_values,
            list(self.weight_size)
        )
        # pass thourgh self weight
        print("input", input.shape)
        print("embed", sparse_embed_weight.shape)
        print(sparse_embed_weight)
        output = torch.sparse.mm(sparse_embed_weight, input.t()).t()
        # concat output of embed weight and input
        input = torch.cat([input, output], dim=1)

        return input


class ExpandingLinear(nn.Module):
    def __init__(self, weight: torch.sparse.FloatTensor, bias: torch.sparse.FloatTensor):
        super(ExpandingLinear, self).__init__()

        # sparse weight
        self.weight_indices = weight.coalesce().indices()
        self.weight_values = nn.Parameter(weight.coalesce().values())
        self.weight_size = list(weight.coalesce().size())

        self.embed_linears = []

        # sparse bias
        # TODO think about bias representation
        self.bias_indices = bias.coalesce().indices()
        self.bias_values = nn.Parameter(bias.coalesce().values())
        self.bias_size = list(bias.coalesce().size())

        self.last_iteration = -1

    def replace(self, child, parent, iteration):
        if iteration > self.last_iteration:
            self.last_iteration = iteration
            self.embed_linears += [EmbedLinear(self.weight_size[1])]
        # mask of edge to remove in MAIN weight
        matches = (self.weight_indices[0] == child) & \
                  (self.weight_indices[1] == parent)
        index_to_remove = matches.nonzero(as_tuple=True)[0]  # index of edge to remove in MAIN weight

        self.weight_indices = self.weight_indices[:,
                              torch.logical_not(matches)]  # remove edge from MAIN weight by masking

        # concated input from embed weight will pass through last vertices in MAIN layer
        max_parent = self.weight_indices[1].max() + 1  # increase number of nodes in "input" of MAIN layer
        self.weight_indices = torch.cat([self.weight_indices, torch.tensor([[child, max_parent]]).t()],
                                        dim=1)  # add new edge to MAIN weight

        mask = torch.ones_like(self.weight_values, dtype=torch.bool)
        mask[index_to_remove] = False
        self.weight_values = nn.Parameter(
            self.weight_values[mask])  # remove value of deleted edge from MAIN value list
        # todo smart weight generation
        self.weight_values.data = torch.cat(
            [self.weight_values.data, torch.rand(1)])  # add new random weight to end of MAIN value list

        self.weight_size[1] += 1  # increase number of nodes in "input" of MAIN layer

        # add new edge to embed weight
        # where self.child_counter is number of nodes in embed weight
        # and parent is number of input node
        self.embed_linears[iteration].replace(child, parent)

    def forward(self, input):
        for i in range(self.last_iteration + 1):
            input = self.embed_linears[i](input)

        sparse_weight = torch.sparse.FloatTensor(self.weight_indices, self.weight_values, self.weight_size)
        sparse_bias = torch.sparse.FloatTensor(self.bias_indices, self.bias_values, self.bias_size).to_dense()
        output = torch.sparse.mm(sparse_weight, input.t()).t()
        output += sparse_bias.unsqueeze(0)

        return output
