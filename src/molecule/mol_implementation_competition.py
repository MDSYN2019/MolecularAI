# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import logging
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

from mol_functions import (
    canon_polymer_smiles,
    load_ffv_dataset,
    load_official,
    load_tc_dataset,
    load_tg_dataset,
    mol_to_graph,
    rdkit_globals,
    randomize_smiles,
    advanced_smiles_to_graph

)
from mol_losses import ContestWMAE
from mol_multitask_utils import (
    attach_multitask_targets,
    compute_scalers,
    counts_from_loader,
    ranges_from_minmax_dict,
    split_df,
)
from mol_torch_gnn_implementation import  train_and_evaluate_edge
from mol_models import GINELayerUpgraded

# ---------------------------
# Configuration
# ---------------------------

PROPERTIES = ["Tg", "FFV", "Tc", "Density", "Rg"]
NUM_TASKS = len(PROPERTIES)
PATH = "/home/sang/Desktop/neurips-open-polymer-prediction-2025/molecule/"
TRAIN_CSV = f"{PATH}/train.csv"
TEST_CSV = f"{PATH}/test.csv"

SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 4
HIDDEN = 384
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4

MINMAX_DICT = {
    "Tg": [-148.0297376, 472.25],
    "FFV": [0.2269924, 0.77709707],
    "Tc": [0.0465, 0.524],
    "Density": [0.748691234, 1.840998909],
    "Rg": [9.7283551, 34.672905605],
}
NUMERIC_COLS = ["Tg", "FFV", "Tc", "Density", "Rg"]

logger = logging.getLogger(__name__)


def add_weight_decay(model: torch.nn.Module, weight_decay: float):
    """
    weight decay function to add into our pytorch function
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim == 1
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "bn" in name.lower()
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def ranges_from_frame(
    df: pd.DataFrame,
    properties: list[str],
    device: torch.device,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> torch.Tensor:
    """
    Per property in the pandas table, return the tensor containing the ranges
    for each value  - taken from the default values in the function, we are returning
    the 1%-percentile and 99th percentile, and getting the absolute value of the difference
    between them
    """
    ranges = []
    for prop in properties:
        series = pd.to_numeric(df[prop], errors="coerce").dropna() # convert column to numeric values
        if len(series) == 0:
            ranges.append(1.0)
            continue
        lo, hi = np.percentile(series, [p_low, p_high]) # compute the 1% 99% value of the series
        ranges.append(max(float(hi - lo), 1e-8)) 
    return torch.tensor(ranges, dtype=torch.float32, device=device)

def attach_u_features(df: pd.DataFrame, u_scaler: StandardScaler) -> pd.DataFrame:
    """
    Attaching the 'universal' features as listed in the rdkit list

    at the moment, these are on a general molecule level, rather than a node level or the edge level
    """
    df = df.copy().reset_index(drop=True)
    df["graph"] = df["graph"].astype(object)

    u_vectors = []
    for smiles in df["SMILES"]:
        values = rdkit_globals(smiles) # compute vector of RD_FEATURES 
        values = u_scaler.transform(values.reshape(1, -1))[0] # transform the features with the standardscaler to normalize
        u_vectors.append(torch.tensor(values, dtype=torch.float32)) # convert to pytorch tensor and append to output list 
    df["u"] = u_vectors # insert into the pandas table 

    for idx, graph in df["graph"].items():
        graph.u = df.at[idx, "u"].unsqueeze(0) # 
    return df

def make_loaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
):
    """
    Convert the pandas table into train, val and test loaders and shuffle the data before returning
    """
    use_cuda = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(df_train["graph"].tolist(), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(df_val["graph"].tolist(), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(df_test["graph"].tolist(), shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def build_training_frame() -> pd.DataFrame:
    """
    build training data 
    """
    official = load_official(TRAIN_CSV)
    supplemental = [
        load_tc_dataset("train_supplement/dataset1.csv"),
        load_tg_dataset("train_supplement/dataset3.csv"),
        load_ffv_dataset("train_supplement/dataset4.csv", header_has_names=False),
    ]
    # combine the official data and the supplementary data that was provided 
    combined = pd.concat([official] + supplemental, ignore_index=True, sort=False)
    
    for col in NUMERIC_COLS: # loop through the numeric columns 
        combined[col] = pd.to_numeric(combined[col], errors="coerce") # convert each column to numeric columns

    combined["SMILES"] = combined["SMILES"].map(canon_polymer_smiles) # standardize the smiles
    combined = combined[combined["SMILES"].notna()].copy() # generate a copy with non-na data

    # clip data within a range 
    combined.loc[combined["FFV"].notna(), "FFV"] = combined["FFV"].clip(0.0, 1.0)
    combined.loc[combined["Density"].notna(), "Density"] = combined["Density"].clip(
        lower=0.3,
        upper=3.0,
    )

    agg = {col: "median" for col in NUMERIC_COLS}
    combined = combined.groupby("SMILES", as_index=False, sort=False).agg(agg)
    combined["SMILES"] = combined["SMILES"].apply(randomize_smiles)
    combined["graph"] = combined["SMILES"].apply(advanced_smiles_to_graph) # create the graph column representation 
    return combined


def build_test_graph_loader(test_df: pd.DataFrame, u_scaler: StandardScaler) -> tuple[DataLoader, list[int]]:
    """
    """
    test_graphs, test_ids = [], []
    for _, row in test_df.iterrows():
        smiles = str(row["SMILES"]).strip()
        try:
            graph = mol_to_graph(smiles)
            values = rdkit_globals(smiles)
            values = u_scaler.transform(values.reshape(1, -1))[0]
            graph.u = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
            test_graphs.append(graph)
            test_ids.append(int(row["id"]))
        except Exception as exc:
            logger.info(f"[WARN] failed SMILES idx={row.name} :: {exc}")

    return DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False), test_ids


def predict(model, loader: DataLoader, device: torch.device, inv_std):
    """
    prediction with the trained model 
    """
    model.to(device)
    model.eval()
    chunks = []

    with torch.no_grad():  # turn off autograd
        for data in loader:
            data = data.to(device)
            data.x = data.x.float()
            if getattr(data, "edge_attr", None) is not None:
                data.edge_attr = data.edge_attr.float()

            out = model(
                data.x, # node featues                 
                data.edge_index, # edge indices 
                data.edge_attr, # edge attribute 
                batch=data.batch, 
                u=getattr(data, "u", None),
            )
            chunks.append(inv_std(out).detach().cpu())

    return torch.cat(chunks, dim=0).numpy()


def main() -> None:
    train_df = build_training_frame() # this now utilizes the advanced_smiles_to_features - which implemented node level and bond level features
    _ = pd.read_csv(TEST_CSV)

    u_train = np.stack([rdkit_globals(s) for s in train_df["SMILES"]], axis=0) # global level features for each molecule also added 
    u_scaler = StandardScaler().fit(u_train) # scale the global feature for each molecule 

    df_train, df_val, df_test = split_df(train_df, train=0.8, val=0.1, seed=SEED)
    scalers = compute_scalers(df_train, PROPERTIES)

    # need a better explanation of what this is doing
    df_train = attach_multitask_targets(df_train, scalers, PROPERTIES)
    df_val = attach_multitask_targets(df_val, scalers, PROPERTIES)
    df_test = attach_multitask_targets(df_test, scalers, PROPERTIES)

    df_train = attach_u_features(df_train, u_scaler)
    df_val = attach_u_features(df_val, u_scaler)
    df_test = attach_u_features(df_test, u_scaler)

    train_loader, val_loader, test_loader = make_loaders(df_train, df_val, df_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    _ = ranges_from_minmax_dict(MINMAX_DICT, PROPERTIES, device)

    # 
    ranges_train = ranges_from_frame(df_train, PROPERTIES, device)
    ranges_val = ranges_from_frame(df_val, PROPERTIES, device)
    ranges_test = ranges_from_frame(df_test, PROPERTIES, device)

    n_train = counts_from_loader(train_loader, NUM_TASKS, device)

    mu = torch.tensor([scalers[p][0] for p in PROPERTIES], dtype=torch.float32, device=device)
    sd = torch.tensor([scalers[p][1] for p in PROPERTIES], dtype=torch.float32, device=device)

    def inv_std(z: torch.Tensor) -> torch.Tensor:
        return z * sd.to(z.device) + mu.to(z.device)

    criterion_train = ContestWMAE(ranges_train, n_train, inverse_transform=inv_std).to(device)
    criterion_eval = ContestWMAE(ranges_val, n_train, inverse_transform=inv_std).to(device)
    criterion_test = ContestWMAE(ranges_test, n_train, inverse_transform=inv_std).to(device)

    sample = next(iter(train_loader))
    edge_attr = getattr(sample, "edge_attr", None)

    if edge_attr is None:
        raise RuntimeError(
            "GINELayer expects edge features; got None. "
            "Supply edge_attr or switch to a non-edge GIN."
        )

    model = GINELayerUpgraded(
        in_channels=sample.num_node_features,
        edge_dim=edge_attr.size(-1) if edge_attr.dim() >= 2 else 1,
        out_channels=NUM_TASKS,
        hidden_channels=HIDDEN,
        num_layers=8,
        dropout=0.25,
        pooling="meanmax",
        use_gru=False,
        use_residual=True,
        norm="batch",
        global_feat_dim=sample.u.size(1) if hasattr(sample, "u") else 0,
    )

    optimizer = torch.optim.AdamW(
        add_weight_decay(model, WEIGHT_DECAY),
        lr=LR,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        threshold=3e-4,
        cooldown=3,
        min_lr=1e-6,
        verbose=True,
    )

    results = train_and_evaluate_edge(
        model=model,
        optimizer=optimizer,
        criterion_train=criterion_train,
        criterion_eval=criterion_eval,
        criterion_test=criterion_test,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        early_stopping=True,
        eval_every=5,
        scheduler=scheduler,
    )

    torch.save(model.state_dict(), "model_best.pt")
    logger.info(
        "Done. Test loss:",
        results["test_losses"],
        "MAE:",
        results["test_mae"],
        "R2:",
        results["test_r2"],
    )

    preds_z = results["test_preds"]
    targets_z = results["test_targets"]
    mask = results["test_mask"].float()

    mu_cpu = torch.tensor([scalers[p][0] for p in PROPERTIES], dtype=torch.float32)
    sd_cpu = torch.tensor([scalers[p][1] for p in PROPERTIES], dtype=torch.float32)

    preds_orig = preds_z * sd_cpu + mu_cpu
    targets_orig = targets_z * sd_cpu + mu_cpu
    per_prop_mae = ((preds_orig - targets_orig).abs() * mask).sum(0) / mask.sum(0).clamp_min(1.0)

    contest_wmae = criterion_test(
        preds_z.to(criterion_test.weights.device),
        targets_z.to(criterion_test.weights.device),
        mask.to(criterion_test.weights.device),
    ).item()

    
    logger.info("Per-property MAE:", per_prop_mae.tolist())
    logger.info("Contest wMAE:", contest_wmae)

    test_df = pd.read_csv(TEST_CSV)
    test_loader_only, test_ids = build_test_graph_loader(test_df, u_scaler)

    pred_mat = predict(model, test_loader_only, device, inv_std)
    submission = pd.DataFrame(pred_mat, columns=PROPERTIES)
    submission.insert(0, "id", test_ids)
    final_output = pd.merge(test_df, submission, on="id")
    return final_output


if __name__ == "__main__":
    main()
