import torch
from torch import nn

from .model import ExpandingLinear


def dense_to_sparse(dense_tensor: torch.Tensor, device='cpu') -> torch.Tensor:
    """
    Converts a dense tensor to a sparse tensor, with the option to specify the device.

    Args:
        dense_tensor (torch.Tensor): Dense tensor to convert.
        device (str): Device where the sparse tensor should reside.

    Returns:
        torch.sparse_coo_tensor: Sparse representation of the dense tensor.
    """
    indices = dense_tensor.nonzero(as_tuple=True)
    values = dense_tensor[indices]
    indices = torch.stack(indices).to(device)

    sparse_tensor = torch.sparse_coo_tensor(indices, values, dense_tensor.size(), device=device)
    return sparse_tensor


def convert_dense_to_sparse_network(model: nn.Module, device='cpu') -> nn.Module:
    """
    Converts a given dense neural network model to a sparse neural network model.

    This function recursively iterates through the given model and replaces all instances of
    `nn.Linear` layers with `ExpandingLinear` layers.

    Args:
        model (nn.Module): The dense neural network model to be converted.
        device (str): Device where the sparse model and its tensors should reside.

    Returns:
        nn.Module: A new neural network model with sparse layers.
    """
    new_model = model.__class__()

    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            sparse_weight = dense_to_sparse(module.weight.data, device=device)
            sparse_bias = dense_to_sparse(module.bias.data, device=device)

            setattr(new_model, name, ExpandingLinear(sparse_weight, sparse_bias, device=device))
        else:
            setattr(new_model, name, convert_dense_to_sparse_network(module, device=device))
    return new_model


def get_model_last_layer(model):
    for layer in reversed(list(model.modules())):
        if isinstance(layer, (nn.Linear, ExpandingLinear)):
            return layer
    return None


def freeze_all_but_last(model: nn.Module):
    last_layer_params = None

    for name, param in reversed(list(model.named_parameters())):
        if 'weight' in name or 'bias' in name:
            last_layer_params = param
            break

    for param in model.parameters():
        param.requires_grad_(False)

    if isinstance(last_layer_params, ExpandingLinear):
        last_layer_params.freeze_embeds()


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad_(True)

        if isinstance(param, ExpandingLinear):
            param.unfreeze_embeds()
