import pandas as pd
import torch
from torch_geometric.data import Data

from molecule.mol_losses import ContestWMAE
from molecule.mol_multitask_utils import (
    attach_multitask_targets,
    compute_scalers,
    counts_from_loader,
    ranges_from_minmax_dict,
    split_df,
)


def test_contest_wmae_matches_manual_calculation():
    ranges = torch.tensor([10.0, 20.0])
    counts = torch.tensor([4.0, 9.0])
    criterion = ContestWMAE(ranges=ranges, counts=counts)

    pred = torch.tensor([[2.0, 5.0], [1.0, 1.0]])
    target = torch.tensor([[1.0, 2.0], [1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    loss = criterion(pred, target, mask)

    inv_ranges = 1.0 / ranges
    inv_sqrt_counts = 1.0 / torch.sqrt(counts)
    alpha = ranges.numel() / inv_sqrt_counts.sum()
    weights = inv_ranges * inv_sqrt_counts * alpha

    expected = (((pred - target).abs() * mask) * weights).sum(dim=1).mean()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_multitask_targets_and_scalers_attach_to_graphs():
    df = pd.DataFrame(
        {
            "graph": [Data(x=torch.randn(2, 3)), Data(x=torch.randn(2, 3))],
            "Tg": [100.0, 110.0],
            "FFV": [0.2, float("nan")],
        }
    )

    scalers = compute_scalers(df, ["Tg", "FFV"])
    out = attach_multitask_targets(df, scalers, ["Tg", "FFV"])

    g0 = out.iloc[0]["graph"]
    g1 = out.iloc[1]["graph"]

    assert g0.y.shape[0] == 2
    assert g0.y_mask.tolist() == [1.0, 1.0]
    assert g1.y_mask.tolist() == [1.0, 0.0]


def test_split_ranges_and_counts_utilities():
    df = pd.DataFrame({"v": list(range(10))})
    train_df, val_df, test_df = split_df(df, train=0.6, val=0.2, seed=7)
    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    ranges = ranges_from_minmax_dict(
        {"Tg": [10.0, 40.0], "FFV": [0.1, 0.5]},
        ["Tg", "FFV"],
        torch.device("cpu"),
    )
    assert torch.allclose(ranges, torch.tensor([30.0, 0.4]))

    class DummyBatch:
        def __init__(self, mask):
            self.y_mask = mask

    loader = [
        DummyBatch(torch.tensor([[1.0, 0.0], [1.0, 1.0]])),
        DummyBatch(torch.tensor([[0.0, 1.0]])),
    ]
    counts = counts_from_loader(loader, num_tasks=2, device=torch.device("cpu"))
    assert torch.equal(counts, torch.tensor([2.0, 2.0]))
