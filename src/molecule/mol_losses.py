import torch
import torch.nn as nn


class ContestWMAE(nn.Module):
    """
    Contest weighted-MAE used by the polymer multitask workflow.

    loss = mean_over_batch( sum_i w_i * |pred_i - target_i| * mask_i )
    with:
      w_i = (1 / r_i) * ( K / sqrt(n_i) ) / sum_j(1 / sqrt(n_j))

    Parameters
    ----------
    ranges:
        Tensor [T] with task ranges in original units (max - min).
    counts:
        Tensor [T] with task-specific label counts.
    inverse_transform:
        Optional callable mapping standardized targets back to original units.
    eps:
        Numerical stability term.
    """

    def __init__(
        self,
        ranges: torch.Tensor,
        counts: torch.Tensor,
        inverse_transform=None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        num_tasks = ranges.numel()
        inv_ranges = (1.0 / ranges.clamp_min(eps)).float()
        inv_sqrt_counts = (1.0 / torch.sqrt(counts.clamp_min(1.0))).float()
        alpha = num_tasks / inv_sqrt_counts.sum().clamp_min(eps)
        weights = inv_ranges * inv_sqrt_counts * alpha

        self.register_buffer("weights", weights)
        self.inverse_transform = inverse_transform

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if self.inverse_transform is not None:
            pred = self.inverse_transform(pred)
            target = self.inverse_transform(target)

        abs_err = (pred - target).abs() * mask
        per_sample = (abs_err * self.weights).sum(dim=1)
        return per_sample.mean()
