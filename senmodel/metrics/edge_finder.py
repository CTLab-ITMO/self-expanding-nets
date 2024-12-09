import torch

from .nonlinearity_metrics import NonlinearityMetric
from ..model.utils import get_model_last_layer


class EdgeFinder:
    def __init__(self, metric: NonlinearityMetric, dataloader, device=torch.device('cpu')):
        self.metric = metric
        self.dataloader = dataloader
        self.device = device

    def calculate_edge_metric_for_dataloader(self, model):
        accumulated_grads = None
        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)

            metric = self.metric.calculate(model, data, target)

            if accumulated_grads is None:
                accumulated_grads = torch.zeros_like(metric).to(self.device)

            accumulated_grads += metric

        return accumulated_grads / len(self.dataloader)

    def choose_edges_top_k(self, model, top_k: int):
        assert top_k > 0
        avg_metric = self.calculate_edge_metric_for_dataloader(model)
        sorted_indices = torch.argsort(avg_metric, descending=True)
        last_layer = get_model_last_layer(model)
        return last_layer.weight_indices[:, sorted_indices[:top_k]]

    def choose_edges_top_percent(self, model, percent: float):
        assert 0 < percent <= 1
        avg_metric = self.calculate_edge_metric_for_dataloader(model)
        k = int(percent * avg_metric.numel())
        sorted_indices = torch.argsort(avg_metric, descending=True)
        last_layer = get_model_last_layer(model)
        return last_layer.weight_indices[:, sorted_indices[:k]]

    def choose_edges_threshold(self, model, threshold):
        assert 0 < threshold <= 1
        avg_metric = self.calculate_edge_metric_for_dataloader(model)
        mask = avg_metric > threshold
        last_layer = get_model_last_layer(model)
        return last_layer.weight_indices[:, mask.nonzero(as_tuple=True)[0]]
