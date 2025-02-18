from abc import abstractmethod, ABC

import torch
from torch import nn
from random import random


class SparseModule(ABC, nn.Module):
    def __init__(self, weight_size, device='cpu'):
        super(SparseModule, self).__init__()
        self.weight_indices = torch.empty(2, 0, dtype=torch.long, device=device)
        self.weight_values = nn.Parameter(torch.empty(0, device=device))
        self.weight_size = list(weight_size)
        self.device = device

        self.activation = nn.Tanh()

    def add_edge(self, child, parent, original_weight, new=True):
        new_edge = torch.tensor([[child, parent]], dtype=torch.long, device=self.device).t()
        self.weight_indices = torch.cat([self.weight_indices, new_edge], dim=1)
        new_weight = (
            torch.tensor(
                (1.0 + random() / 100) if new else original_weight,
                device=self.device).unsqueeze(0)
        )
        # new_weight = torch.ones(1, device=self.device)
        # nn.init.uniform_(new_weight)
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
    def __init__(self, weight_size, device='cpu'):
        super(EmbedLinear, self).__init__([0, weight_size], device=device)
        self.child_counter = 0
        self.device = device

    def replace(self, child, parent, original_weight=1.):
        # matches = (self.weight_indices[0] == child) & (self.weight_indices[1] == parent)
        #
        # assert torch.any(matches), "Edge must extist"
        #
        # original_weight = self.weight_values[matches].item()
        self.add_edge(self.child_counter, parent, original_weight=original_weight, new=False)
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

        self.count_replaces = [self.weight_indices.size(1)]

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

        original_weight = self.weight_values[matches].item()
        max_parent = self.weight_indices[1].max().item() + 1  # before deleting edge

        self.weight_indices = self.weight_indices[:, ~matches]
        self.weight_values = nn.Parameter(self.weight_values[~matches])

        self.add_edge(child, max_parent, original_weight)

        self.weight_size[1] += 1
        self.embed_linears[self.current_iteration].replace(child, parent, original_weight)

    def replace_many(self, children, parents):
        replaced_count = len(children)  
        self.count_replaces.append(replaced_count)
        
        if len(children) and len(parents):
            self.current_iteration += 1
        
        super().replace_many(children, parents)

    def freeze_embeds(self, len_choose):
        # freeze_all_but_last
        with torch.no_grad():
            if self.embed_linears:
                # print("weight grads")
                # print(model.fc1.weight_values.grad)

                for i in range(len(self.embed_linears) - 1):
                    self.embed_linears[i].weight_values.grad.zero_()
                for i in range(len(self.weight_values) - len_choose):
                    self.weight_values.grad[i] = 0

                # print("weight grads zero")
                # print(model.fc1.weight_values.grad)

    def unfreeze_embeds(self):
        # Разморозить все веса в embed_linears
        for embed_linear in self.embed_linears:
            for param in embed_linear.parameters():
                param.requires_grad = True

        # Разморозить все веса в weight_values
        if hasattr(self, "weight_values"):
            for param in self.weight_values:
                param.requires_grad = True

    def forward(self, input):
        # Применяем все EmbedLinear слои
        for embed_linear in self.embed_linears:
            input = embed_linear(input)

        # Создаём разреженную матрицу весов с учётом маски
        masked_weight_values = self.weight_values * self.weight_mask if hasattr(self,
                                                                                'weight_mask') else self.weight_values
        sparse_weight = torch.sparse_coo_tensor(self.weight_indices, masked_weight_values, self.weight_size,
                                                device=self.device)

        # Применяем маску к смещениям (bias), если она есть
        masked_bias_values = self.bias_values * self.bias_mask if hasattr(self, 'bias_mask') else self.bias_values
        sparse_bias = torch.sparse_coo_tensor(self.bias_indices, masked_bias_values, self.bias_size,
                                              device=self.device).to_dense()

        # Вычисляем линейное преобразование
        output = torch.sparse.mm(sparse_weight, input.t()).t()
        output += sparse_bias.unsqueeze(0)

        return output
