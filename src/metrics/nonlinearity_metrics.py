from abc import abstractmethod, ABC
import torch


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

        last_layer = model

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

        last_layer = model
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


