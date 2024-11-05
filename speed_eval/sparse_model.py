import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self, all_sizes, activation=nn.ReLU):
        super(MyModel, self).__init__()
        input_size, *hidden_sizes, output_size = all_sizes
        layers = []
        current_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(activation())
            current_size = hidden_size
        layers.append(nn.Linear(current_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features).to_sparse())
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        output = torch.sparse.mm(self.weight, input.t()).t()
        if self.bias is not None:
            output += self.bias
        return output


def convert_sparse_weights_to_dense(model):
    for module in model.modules():
        if isinstance(module, SparseLinear):
            module.weight = nn.Parameter(module.weight.to_dense()) 


def convert_dense_weights_to_sparse(model):
    for module in model.modules():
        if isinstance(module, SparseLinear):
            module.weight = nn.Parameter(module.weight.to_sparse()) 


def convert_layer(layer):
    if isinstance(layer, nn.Linear):
        sparse_layer = SparseLinear(layer.in_features, layer.out_features, bias=layer.bias is not None)
        sparse_layer.weight = nn.Parameter(layer.weight.to_sparse())
        if layer.bias is not None:
            sparse_layer.bias = nn.Parameter(layer.bias)
        return sparse_layer
    else:
        return layer
        

def convert_to_sparse(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            for idx, layer in enumerate(module):
                module[idx] = convert_layer(layer)
        else:
            setattr(model, name, convert_layer(module))
    return model
