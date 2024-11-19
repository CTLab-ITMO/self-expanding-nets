from abc import abstractmethod, ABC
import torch
import torch.nn as nn
import torch.nn.functional as F


class NonlinearityMetric(ABC):
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    @staticmethod
    def _get_last_sparse_layer(model):
        # if not isinstance(model, SparseRecursiveLinear):
        #     raise ValueError("Expected a SparseRecursiveLinear model.")
        return model.sparse_linear

    @abstractmethod
    def calculate(self, model, X_arr, y_arr):
        pass

    def __call__(self, model, X_arr, y_arr, top_k: int = 1):
        model.eval()
        last_layer = self._get_last_sparse_layer(model)
        values = self.calculate(model, X_arr, y_arr)

        # Индексы, отсортированные по убыванию значений метрики
        sorted_indices = torch.argsort(values, descending=True)

        # Возвращаем топ k индексов из last_layer.weight_indices
        top_k_indices = last_layer.weight_indices[sorted_indices[:top_k]]
        return top_k_indices


# Метрика 1: Средний градиент для каждого ребра
class GradientMeanEdgeMetric(NonlinearityMetric):
    def calculate(self, model, X_arr, y_arr):
        model.eval()
        model.zero_grad()

        y_pred = model(X_arr).squeeze()
        loss = self.loss_fn(y_pred, y_arr)
        loss.backward()

        last_layer = self._get_last_sparse_layer(model)

        # Градиенты для разреженных весов
        edge_gradients = last_layer.weight_values.grad.abs()
        model.zero_grad()
        return edge_gradients


# Метрика 3: Чувствительность к возмущению для каждого ребра
class PerturbationSensitivityEdgeMetric(NonlinearityMetric):
    def __init__(self, loss_fn, epsilon=1e-2):
        super().__init__(loss_fn)
        self.epsilon = epsilon

    def calculate(self, model, X_arr, y_arr):
        model.eval()

        # Оригинальный вывод модели
        original_output = model(X_arr).detach()

        last_layer = self._get_last_sparse_layer(model)
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

        return sensitivities
