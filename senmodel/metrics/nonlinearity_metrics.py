import copy
from abc import abstractmethod, ABC

import torch
from torch import nn

from ..model.model import ExpandingLinear
from ..model.utils import get_model_last_layer, unfreeze_all


# def get_weights(model, layers, masks):
#     for layer in layers:
#         weights = model.__getatr__(layer)
#     last_layer = get_model_last_layer(model)
#     if len_choose is None:
#         return last_layer.weight_values
#     if len_choose == 0:
#         return torch.tensor([]) 
    
#     return last_layer.weight_values[-len_choose:] 

# def get_weights(model, len_choose):
#     last_layer = get_model_last_layer(model)
#     if len_choose is None:
#         return last_layer.weight_values
#     if len_choose == 0:
#         return torch.tensor([]) 
    
#     return last_layer.weight_values[-len_choose:] 


# def get_weights_grad(model, len_choose):
#     last_layer = get_model_last_layer(model)
#     grad = last_layer.weight_values.grad
    
#     if len_choose is None:
#         return grad
#     if len_choose == 0:
#         return torch.tensor([]) 
#     return grad[-len_choose:] 


class NonlinearityMetric(ABC):
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    @abstractmethod
    def calculate(self, model, layer_name, mask, X_arr, y_arr):
        pass

class AbsGradientEdgeMetric(NonlinearityMetric):
    def calculate(self, model, layer_name, mask, X_arr, y_arr):
        model = copy.deepcopy(model)
        unfreeze_all(model)
        model.eval()
        model.zero_grad()

        y_pred = model(X_arr).squeeze()
        loss = self.loss_fn(y_pred, y_arr)
        loss.backward()

        layer = model.__getattr__(layer_name)
        edge_gradients = layer.weight_values.grad[mask].abs()
        model.zero_grad()
        return edge_gradients

class ReversedAbsGradientEdgeMetric(NonlinearityMetric):
    def calculate(self, model, layer_name, mask, X_arr, y_arr):
        model = copy.deepcopy(model)
        unfreeze_all(model)
        model.eval()
        model.zero_grad()

        y_pred = model(X_arr).squeeze()
        loss = self.loss_fn(y_pred, y_arr)
        loss.backward()

        layer = model.__getattr__(layer_name)
        edge_gradients = 1 / (layer.weight_values.grad[mask].abs() + 1e-8)
        model.zero_grad()
        return edge_gradients


# class SNIPMetric(NonlinearityMetric):
#     def calculate(self, model, X_arr, y_arr, len_choose): #todo len_choose
#         model = copy.deepcopy(model)
#         unfreeze_all(model)
#         model.eval()

#         for layer in model.modules():
#             if isinstance(layer, (nn.Linear, ExpandingLinear)):
#                 w = layer.weight if not isinstance(layer, ExpandingLinear) else layer.weight_values
#                 layer.weight_mask = nn.Parameter(torch.ones_like(w))
#                 if isinstance(layer, ExpandingLinear):
#                     nn.init.normal_(w, mean=0, std=0.01)
#                 else:
#                     nn.init.xavier_normal_(w)
#                 w.requires_grad = False

#         model.zero_grad()
#         outputs = model(X_arr).squeeze()
#         loss = self.loss_fn(outputs, y_arr)
#         loss.backward()

#         edge_gradients = get_model_last_layer(model).weight_mask.grad.abs()
#         model.zero_grad()
#         return edge_gradients


class MagnitudeL1Metric(NonlinearityMetric):
    def calculate(self, model, layer_name, mask, X_arr=None, y_arr=None):
        layer = model.__getattr__(layer_name)
        return layer.weight_values[mask].abs()


class MagnitudeL2Metric(NonlinearityMetric):
    def calculate(self, model, layer_name, mask, X_arr=None, y_arr=None):
        layer = model.__getattr__(layer_name)
        return torch.pow(layer.weight_values[mask], 2)


# class PerturbationSensitivityEdgeMetric(NonlinearityMetric):
#     def __init__(self, loss_fn, epsilon=1e-2):
#         super().__init__(loss_fn)
#         self.epsilon = epsilon

#     def calculate(self, model, X_arr, y_arr, len_choose): #todo len_choose
#         model.eval()
#         original_output = model(X_arr).detach()
#         last_layer = get_model_last_layer(model)
#         sensitivities = torch.zeros_like(last_layer.weight_values)

#         for idx in range(last_layer.weight_values.size(0)):
#             with torch.no_grad():
#                 original_value = last_layer.weight_values[idx].item()
#                 last_layer.weight_values[idx] += self.epsilon
#                 perturbed_output = model(X_arr)
#                 sensitivity = (perturbed_output - original_output).abs().mean().item()
#                 sensitivities[idx] = sensitivity
#                 last_layer.weight_values[idx] = original_value
#         return sensitivities
