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
        threshold=0.15,
        max_to_choose=None,
    ):
        self.metric = metric
        self.dataloader = dataloader
        self.device = device
        self.aggregation_mode = aggregation_mode
        assert aggregation_mode in [
            "mean",
            "variance",
        ], "Aggregation mode must be 'mean' or 'variance'."
        
        self.threshold = threshold
        self.max_to_choose = max_to_choose

    def calculate_edge_metric_for_dataloader(self, model, layer, embed=False):
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

        return aggregated

    def choose_edges(self, model, layer, embed=False, max_limit=False):
        values = self.calculate_edge_metric_for_dataloader(
            model=model, layer=layer, embed=embed
        )

        norm_values = (values - values.min()) / (values.max() - values.min())
        
        mask = norm_values <= self.threshold
        
        final_indices = torch.nonzero(mask).squeeze()

        if max_limit and self.max_to_choose is not None:
            final_indices = final_indices[:self.max_to_choose]

        res = layer.weight_indices[:, final_indices]

        return res