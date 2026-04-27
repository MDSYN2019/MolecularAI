from __future__ import annotations

from typing import Iterable

import pandas as pd
import torch
from torch_geometric.data import Data


def split_df(df: pd.DataFrame, train: float = 0.8, val: float = 0.1, seed: int = 42):
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(df), generator=gen)
    n = len(df)
    i_tr = idx[: int(train * n)]
    i_va = idx[int(train * n) : int((train + val) * n)]
    i_te = idx[int((train + val) * n) :]
    return df.iloc[i_tr].copy(), df.iloc[i_va].copy(), df.iloc[i_te].copy()


def compute_scalers(df_train: pd.DataFrame, properties: Iterable[str]) -> dict:
    scalers = {}
    for p in properties:
        s = pd.to_numeric(df_train[p], errors="coerce")
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True)
        if pd.isna(sd) or sd == 0:
            sd = 1.0
        scalers[p] = (float(mu), float(sd))
    return scalers


def attach_multitask_targets(
    df: pd.DataFrame, scalers: dict, properties: Iterable[str]
) -> pd.DataFrame:
    """
    For each molecule/graph,

    collect target values for several properties
    normalize the values
    mark which targets are actually present
    attach both to the graph object 
    
    """
    df = df.copy()
    for i, row in df.iterrows():
        g: Data = row["graph"] 
        y_vals, y_mask = [], []
        
        for p in properties:
            val = row.get(p, None) # get the value of the property 
            if pd.notna(val): # if the value is not missing..
                mu, sd = scalers[p] # get the mean and std for that property 
                y_vals.append((float(val) - mu) / sd) # normalize the value and add to the list of target values 
                y_mask.append(1.0) # mark that this target is present 
            else:
                y_vals.append(0.0) # if the value is missing add a placeholder 
                y_mask.append(0.0) # mark that this target is missing 
        g.y = torch.tensor(y_vals, dtype=torch.float)
        g.y_mask = torch.tensor(y_mask, dtype=torch.float)
        df.at[i, "graph"] = g
    return df


def ranges_from_minmax_dict(
    minmax_dict: dict, properties: Iterable[str], device: torch.device
) -> torch.Tensor:
    mins = torch.tensor(
        [float(minmax_dict[p][0]) for p in properties],
        dtype=torch.float32,
        device=device,
    )
    maxs = torch.tensor(
        [float(minmax_dict[p][1]) for p in properties],
        dtype=torch.float32,
        device=device,
    )
    return (maxs - mins).clamp_min(1e-8)


def counts_from_loader(
    loader, num_tasks: int, device: torch.device
) -> torch.Tensor:
    cnt = torch.zeros(num_tasks, dtype=torch.float32, device=device)
    for batch in loader:
        m = getattr(batch, "y_mask", None)
        if m is None:
            m = (~torch.isnan(batch.y)).float()
        if m.dim() == 1:
            m = m.view(-1, 1)
        cnt += m.sum(dim=0).to(device)
    return cnt.clamp_min(1.0)
