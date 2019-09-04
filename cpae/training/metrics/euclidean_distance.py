from typing import Optional

from overrides import overrides
import torch
import torch.nn as nn

from allennlp.training.metrics.metric import Metric


@Metric.register("eucliean_distance")
class EuclideanDistance(Metric):
    """
    This ``Metric`` calculates the Euclidean Distance between two tensors.
    """
    def __init__(self) -> None:
        self._euclidean_distance = 0.0
        self._total_count = 0.0
        self.pdist = nn.PairwiseDistance(p=2)

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ...).
        gold_labels : ``torch.Tensor``, required.
            A tensor of the same shape as ``predictions``.
        mask: ``torch.Tensor``, optional (default = None).
            A tensor of the same shape as ``predictions``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        euclidean_distances = self.pdist(predictions, gold_labels)
        if mask is not None:
            euclidean_distances *= mask
        self._total_count += gold_labels.numel()
        self._euclidean_distance += torch.sum(euclidean_distances)

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated mean absolute error.
        """
        euclidean_distance = float(self._euclidean_distance)
        # euclidean_distance = float(self._euclidean_distance) / \
        #     float(self._total_count)
        if reset:
            self.reset()
        return euclidean_distance

    @overrides
    def reset(self):
        self._euclidean_distance = 0.0
        self._total_count = 0.0
