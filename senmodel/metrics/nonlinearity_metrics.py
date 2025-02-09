import copy
from abc import abstractmethod, ABC

import torch
from torch import nn

from senmodel.model.utils import get_model_last_layer
from senmodel.model.model import ExpandingLinear


class NonlinearityMetric(ABC):
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    @abstractmethod
    def calculate(self, model, X_arr, y_arr):
        pass


# Метрика 1: Средний градиент для каждого ребра
class GradientMeanEdgeMetric(NonlinearityMetric):
    def calculate(self, model, X_arr, y_arr):
        model.eval()
        model.zero_grad()

        y_pred = model(X_arr).squeeze()
        loss = self.loss_fn(y_pred, y_arr)
        loss.backward()

        last_layer = get_model_last_layer(model)

        # Градиенты для разреженных весов
        edge_gradients = last_layer.weight_values.grad.abs()
        model.zero_grad()

        min_val, max_val = edge_gradients.min(), edge_gradients.max()
        normalized_gradients = (edge_gradients - min_val) / (max_val - min_val + 1e-8)

        return normalized_gradients


class SNIPMetric(NonlinearityMetric):
    def calculate(self, model, X_arr, y_arr):
        model = copy.deepcopy(model)
        model.eval()

        for layer in model.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, ExpandingLinear):
                w = layer.weight if not isinstance(layer, ExpandingLinear) else layer.weight_values

                layer.weight_mask = nn.Parameter(torch.ones_like(w))
                if isinstance(layer, ExpandingLinear):
                    nn.init.normal_(w, mean=0, std=0.01)
                else:
                    nn.init.xavier_normal_(w)
                w.requires_grad = False

        model.zero_grad()
        outputs = model(X_arr).squeeze()
        loss = self.loss_fn(outputs, y_arr)
        loss.backward()

        edge_gradients = get_model_last_layer(model).weight_mask.grad.abs()

        model.zero_grad()

        min_val, max_val = edge_gradients.min(), edge_gradients.max()
        normalized_gradients = (edge_gradients - min_val) / (max_val - min_val + 1e-8)

        return normalized_gradients

class MagnitudeL1Metric(NonlinearityMetric):
    def calculate(self, model, X_arr, y_arr):
        model = copy.deepcopy(model)
        last_layer_weights = get_model_last_layer(model).weight_values.abs()

        min_val, max_val = last_layer_weights.min(), last_layer_weights.max()
        normalized = (last_layer_weights - min_val) / (max_val - min_val + 1e-8)

        return normalized


class MagnitudeL2Metric(NonlinearityMetric):
    def calculate(self, model, X_arr, y_arr):
        model = copy.deepcopy(model)
        last_layer_weights = get_model_last_layer(model).weight_values.pow(2)

        min_val, max_val = last_layer_weights.min(), last_layer_weights.max()
        normalized = (last_layer_weights - min_val) / (max_val - min_val + 1e-8)

        return normalized

# Метрика 3: Чувствительность к возмущению для каждого ребра
class PerturbationSensitivityEdgeMetric(NonlinearityMetric):
    def __init__(self, loss_fn, epsilon=1e-2):
        super().__init__(loss_fn)
        self.epsilon = epsilon

    def calculate(self, model, X_arr, y_arr):
        model.eval()

        # Оригинальный вывод модели
        original_output = model(X_arr).detach()

        last_layer = get_model_last_layer(model)
        sensitivities = torch.zeros_like(last_layer.weight_values)

        # Возмущение каждого веса
        for idx in range(last_layer.weight_values.size(0)):
            with torch.no_grad():
                original_value = last_layer.weight_values[idx].item()
                last_layer.weight_values[idx] += self.epsilon

                # Пересчет модели с возмущением
                perturbed_output = model(X_arr)
                sensitivity = (perturbed_output - original_output).abs().mean().item()
                sensitivities[idx] = sensitivity

                # Восстановление оригинального значения
                last_layer.weight_values[idx] = original_value

        min_val, max_val = sensitivities.min(), sensitivities.max()
        normalized_sensitivities = (sensitivities - min_val) / (max_val - min_val + 1e-8)

        return normalized_sensitivities
