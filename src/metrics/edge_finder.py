import torch

from nonlinearity_metrics import NonlinearityMetric


class EdgeFinder:
    def __init__(self, metric: NonlinearityMetric, dataloader):
        self.metric = metric
        self.dataloader = dataloader

    def calculate_edge_metric_for_dataloader(self, model):
        accumulated_grads = None
        for data, target in self.dataloader:
            data, target = data.to('cpu'), target.to('cpu')

            metric = self.metric.calculate(model, data, target)

            if accumulated_grads is None:
                accumulated_grads = torch.zeros_like(metric).to('cpu')

            accumulated_grads += metric

        return accumulated_grads / len(self.dataloader)

    def choose_edges(self, model, top_k: int):
        avg_metric = self.calculate_edge_metric_for_dataloader(model)
        # Индексы, отсортированные по убыванию значений метрики
        sorted_indices = torch.argsort(avg_metric, descending=True)

        # Возвращаем топ k индексов из last_layer.weight_indices
        top_k_indices = model.weight_indices[:, sorted_indices[:top_k]]

        return top_k_indices
