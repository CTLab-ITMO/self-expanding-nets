from senmodel.model.utils import *

def get_params_amount(model):
    amount = 0
    for _, layer in model.named_children():
        if isinstance(layer, ExpandingLinear):
            for linear in layer.embed_linears:
                amount += linear.weight_values.shape[0]
            amount += layer.weight_values.shape[0]
        elif isinstance(layer, nn.Linear):
            amount += layer.in_features * layer.out_features
    return amount


def get_zero_params_amount(model, eps=1e-8):
    amount = 0
    for _, layer in model.named_children():
        if isinstance(layer, ExpandingLinear):
            for linear in layer.embed_linears:
                amount += linear.weight_values[linear.weight_values.abs() < eps].shape[0]
            amount += layer.weight_values[layer.weight_values.abs() < eps].shape[0]
        elif isinstance(layer, nn.Linear):
            amount += layer.weight[layer.weight.abs() < eps].numel()
    return amount


def get_to_replace_params_amount(ef, model, layers, mask, choose_threshold):
    chosen_edges = 0
    for layer in layers:
        chosen_edges += len(ef.choose_edges_threshold(model, layer, choose_threshold, mask)[0])
    return chosen_edges