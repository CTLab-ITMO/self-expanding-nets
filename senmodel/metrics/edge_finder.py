import torch

from .nonlinearity_metrics import NonlinearityMetric
from ..model.utils import get_model_last_layer


class EdgeFinder:
    def __init__(
        self,
        metric: NonlinearityMetric,
        dataloader,
        device=torch.device("cpu"),
        aggregation_mode="mean",
        max_to_replace=None,
    ):
        self.max_to_replace = max_to_replace
        self.metric = metric
        self.dataloader = dataloader
        self.device = device
        self.aggregation_mode = aggregation_mode
        assert aggregation_mode in [
            "mean",
            "variance",
        ], "Aggregation mode must be 'mean' or 'variance'."

    def calculate_edge_metric_for_dataloader(
        self, model, layer, to_normalise=True, embed=False
    ):
        if self.aggregation_mode == "mean":
            accumulated = None
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                metric = self.metric.calculate(model, layer, data, target, embed)
                if accumulated is None:
                    accumulated = torch.zeros_like(metric).to(self.device)
                accumulated += metric
            aggregated = accumulated / len(self.dataloader)
        elif self.aggregation_mode == "variance":
            sum_ = None
            sum_sq = None
            n = len(self.dataloader)
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                metric = self.metric.calculate(model, layer, data, target, embed)
                if sum_ is None:
                    sum_ = torch.zeros_like(metric).to(self.device)
                    sum_sq = torch.zeros_like(metric).to(self.device)
                sum_ += metric
                sum_sq += metric.pow(2)
            mean = sum_ / n
            aggregated = (sum_sq / n) - mean.pow(2)
        else:
            raise ValueError(f"Unsupported aggregation mode: {self.aggregation_mode}")

        if not to_normalise or aggregated.numel() == 0:
            return aggregated

        min_val = aggregated.min()
        max_val = aggregated.max()
        normalized = (aggregated - min_val) / (max_val - min_val + 1e-8)
        return normalized

    # def choose_edges_top_k(self, model, top_k: int, len_choose: int = None):
    #     assert top_k > 0
    #     avg_metric = self.calculate_edge_metric_for_dataloader(model, len_choose)
    #     sorted_indices = torch.argsort(avg_metric, descending=True)
    #     last_layer = get_model_last_layer(model)
    #     return last_layer.weight_indices[:, sorted_indices[:top_k]]

    # def choose_edges_top_percent(self, model, percent: float, len_choose: int = None):
    #     assert 0 < percent <= 1
    #     avg_metric = self.calculate_edge_metric_for_dataloader(model, len_choose)
    #     k = int(percent * avg_metric.numel())
    #     sorted_indices = torch.argsort(avg_metric, descending=True)
    #     last_layer = get_model_last_layer(model)
    #     return last_layer.weight_indices[:, sorted_indices[:k]]

    def choose_edges_threshold(self, model, layer, threshold, embed=False):
        assert 0 < threshold <= 1
        avg_metric = self.calculate_edge_metric_for_dataloader(
            model=model, layer=layer, to_normalise=True, embed=embed
        )
        mask = avg_metric > threshold
        res = layer.weight_indices[:, mask.nonzero(as_tuple=True)[0]]

        if (
            self.max_to_replace is not None
            and self.max_to_replace > 0
            and len(res[0]) > self.max_to_replace
        ):
            sorted_indices = torch.argsort(avg_metric[mask], descending=True)
            res = res[:, sorted_indices[: self.max_to_replace]]
        return res
