import torch
from torch import nn

from .model import ExpandingLinear, SparseModule


def dense_to_sparse(dense_tensor: torch.Tensor) -> torch.Tensor:
    indices = dense_tensor.nonzero(as_tuple=True)
    values = dense_tensor[indices]
    indices = torch.stack(indices)

    sparse_tensor = torch.sparse_coo_tensor(indices, values, dense_tensor.size())
    return sparse_tensor


def convert_dense_to_sparse_network(model: nn.Module) -> nn.Module:
    """
    Converts a given dense neural network model to a sparse neural network model.

    This function recursively iterate through the given model and replaces all instances of
    `nn.Linear` layers with `SparseLinear` layers

    Args:
        model (nn.Module): The dense neural network model to be converted.

    Returns:
        nn.Module: A new neural network model with sparse layers.
    """
    new_model = model.__class__()

    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            sparse_weight = dense_to_sparse(module.weight.data)
            sparse_bias = dense_to_sparse(module.bias.data)

            setattr(new_model, name, ExpandingLinear(sparse_weight, sparse_bias))
        else:
            setattr(new_model, name, convert_dense_to_sparse_network(module))
    return new_model


def get_model_last_layer(model):
    return model if isinstance(model, SparseModule) else list(model.children())[-1]
