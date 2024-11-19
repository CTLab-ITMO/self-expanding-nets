import torch
import torch.nn as nn
import torch.nn.functional as F


class NonlinearityMetric:
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    @staticmethod
    def _get_last_sparse_layer(model):
        # if not isinstance(model, SparseRecursiveLinear):
        #     raise ValueError("Expected a SparseRecursiveLinear model.")
        return model.sparse_linear

    def calculate(self, model, X_arr, y_arr):
        raise NotImplementedError()


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


# # Метрика 2: Стандартное отклонение активаций для каждого ребра
# class ActivationStdEdgeMetric(NonlinearityMetric):
#     def calculate(self, model, X_arr, y_arr):
#         model.eval()
#
#         # Пропуск через модель
#         activations = model(X_arr)
#
#         last_layer = self._get_last_sparse_layer(model)
#
#         # Для каждого ребра: находим соответствующие активации
#         indices = last_layer.weight_indices
#         print(indices, indices.shape, indices[1])
#         edge_activations = activations[indices[1]] * last_layer.weight_values
#         activation_std_edges = edge_activations.std(dim=0)
#         return activation_std_edges


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


# # Метрика 4: Косинусное расстояние между градиентами для каждого ребра
# class CosineGradientSimilarityEdgeMetric(NonlinearityMetric):
#     def calculate(self, model, X_arr, y_arr):
#         model.eval()
#         model.zero_grad()
#
#         y_pred = model(X_arr).squeeze()
#         loss = self.loss_fn(y_pred, y_arr)
#         loss.backward()
#
#         last_layer = self._get_last_sparse_layer(model)
#
#         # Косинусное сходство между градиентами ребер
#         grad_values = last_layer.weight_values.grad
#         similarities = torch.zeros(grad_values.size(0) - 1)
#
#         for i in range(similarities.size(0)):
#             similarities[i] = F.cosine_similarity(
#                 grad_values[i].unsqueeze(0),
#                 grad_values[i + 1].unsqueeze(0),
#                 dim=1
#             )
#
#         model.zero_grad()
#         return similarities

