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


def convert_dense_to_sparse_network(model: nn.Module, device='cpu', last_linear_module=None) -> nn.Module:
    if last_linear_module is None:
        last_linear_module = get_model_last_layer(model)
    
    new_model = model.__class__()

    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if module is last_linear_module:
                sparse_weight = dense_to_sparse(module.weight.data, device=device)
                sparse_bias = dense_to_sparse(module.bias.data, device=device)
                setattr(new_model, name, ExpandingLinear(sparse_weight, sparse_bias, device=device))
            else:
                setattr(new_model, name, module)
        else:
            setattr(new_model, name, convert_dense_to_sparse_network(module, device=device, last_linear_module=last_linear_module))
    
    return new_model


def get_model_last_layer(model):
    for layer in reversed(list(model.modules())):
        if isinstance(layer, (nn.Linear, ExpandingLinear)):
            return layer
    return None



def freeze_all_but_last(model: nn.Module):
    last_layer_params = get_model_last_layer(model)
    len_choose = last_layer_params.count_replaces

    # for param in model.parameters():
    #     if last_layer_params is not param:
    #         param.requires_grad_(False)

    # if isinstance(last_layer_params, ExpandingLinear):
    #     last_layer_params.freeze_embeds(len_choose)
    with torch.no_grad():
        for i in range(len(last_layer_params.embed_linears) - 1, 0, -1):
            A = last_layer_params.embed_linears[i].weight_indices
            A_norm = A.clone()
            A_norm[1, :] -= len_choose[len(last_layer_params.embed_linears) - i - 1]

            B = last_layer_params.embed_linears[i - 1].weight_indices
            
            last_layer_params.embed_linears[i - 1].weight_values.grad[~torch.isin(B[0, :], A[1, :]).nonzero()].zero_()

        for i in range(len(last_layer_params.weight_values) - len_choose[-1]):
            last_layer_params.weight_values.grad[i] = 0

def freeze_only_last(model: nn.Module, len_choose=0):
    last_layer_params = get_model_last_layer(model)
    last_layer_params.freeze_embeds(len_choose)

def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad_(True)

        if isinstance(param, ExpandingLinear):
            param.unfreeze_embeds()
