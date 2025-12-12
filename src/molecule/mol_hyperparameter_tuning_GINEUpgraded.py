import os, random
import numpy as np
import pandas as pd
import torch
import optuna

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from mol_functions import mol_to_graph, fix_tg_units
from mol_torch_gnn_implementation import GINELayerUpgraded, train_and_evaluate_edge
from mol_implementation_competition import ContestWMAE  # or inline the class from your script

# ---------- CONFIG ----------
PROPERTIES = ["Tg", "FFV", "Tc", "Density", "Rg"]
NUM_TASKS = len(PROPERTIES)
PATH = "/home/sang/Desktop/neurips-open-polymer-prediction-2025/molecule"
TRAIN_CSV = f"{PATH}/train.csv"
SEED = 42

MINMAX_DICT = {
    "Tg": [-148.0297376, 472.25],
    "FFV": [0.2269924, 0.77709707],
    "Tc": [0.0465, 0.524],
    "Density": [0.748691234, 1.840998909],
    "Rg": [9.7283551, 34.672905605],
}

# ---------- UTILS ----------
def set_seed(seed=SEED):
    import torch, numpy as np, random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ranges_from_minmax_dict(minmax_dict, device):
    mins = torch.tensor([float(minmax_dict[p][0]) for p in PROPERTIES], dtype=torch.float32, device=device)
    maxs = torch.tensor([float(minmax_dict[p][1]) for p in PROPERTIES], dtype=torch.float32, device=device)
    return (maxs - mins).clamp_min(1e-8)

def counts_from_loader(loader, num_tasks, device):
    cnt = torch.zeros(num_tasks, dtype=torch.float32, device=device)
    for batch in loader:
        m = getattr(batch, "y_mask", None)
        if m is None:
            m = (~torch.isnan(batch.y)).float()
        if m.dim() == 1:
            m = m.view(-1, 1)
        cnt += m.sum(dim=0).to(device)
    return cnt.clamp_min(1.0)

def compute_scalers(df_train):
    scalers = {}
    for p in PROPERTIES:
        s = pd.to_numeric(df_train[p], errors="coerce")
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True)
        if pd.isna(sd) or sd == 0:
            sd = 1.0
        scalers[p] = (float(mu), float(sd))
    return scalers

def attach_multitask_targets(df, scalers):
    df = df.copy()
    for i, row in df.iterrows():
        g: Data = row["graph"]
        y_vals, y_mask = [], []
        for p in PROPERTIES:
            val = row.get(p, None)
            if pd.notna(val):
                mu, sd = scalers[p]
                y_vals.append((float(val) - mu) / sd)  # z-score
                y_mask.append(1.0)
            else:
                y_vals.append(0.0)
                y_mask.append(0.0)
        g.y = torch.tensor(y_vals, dtype=torch.float)
        g.y_mask = torch.tensor(y_mask, dtype=torch.float)
        df.at[i, "graph"] = g
    return df

def split_df(df, train=0.8, val=0.1, seed=SEED):
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(df), generator=gen)
    n = len(df)
    i_tr = idx[: int(train * n)]
    i_va = idx[int(train * n) : int((train + val) * n)]
    i_te = idx[int((train + val) * n) :]
    return df.iloc[i_tr].copy(), df.iloc[i_va].copy(), df.iloc[i_te].copy()

# ---------- DATA (built once) ----------
set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv(TRAIN_CSV)
train_df = fix_tg_units(train_df)
train_df["graph"] = train_df["SMILES"].apply(mol_to_graph)

df_train, df_val, df_test = split_df(train_df, train=0.8, val=0.1, seed=SEED)
scalers = compute_scalers(df_train)

df_train = attach_multitask_targets(df_train, scalers)
df_val   = attach_multitask_targets(df_val,   scalers)
df_test  = attach_multitask_targets(df_test,  scalers)

mu_t = torch.tensor([scalers[p][0] for p in PROPERTIES], dtype=torch.float32, device=device)
sd_t = torch.tensor([scalers[p][1] for p in PROPERTIES], dtype=torch.float32, device=device)
def inv_std(z, mu=mu_t, sd=sd_t):
    return z * sd.to(z.device) + mu.to(z.device)

ranges_all = ranges_from_minmax_dict(MINMAX_DICT, device)

# ---------- OPTUNA OBJECTIVE ----------
def objective(trial: optuna.Trial) -> float:
    set_seed(SEED + trial.number)  # small jitter per trial

    # Search space
    hidden = trial.suggest_categorical("hidden", [128, 256, 512])
    num_layers = trial.suggest_categorical("num_layers", [3, 4, 5])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    epochs = trial.suggest_int("epochs", 80, 160, 220)  # short runs; early stop will keep it efficient
    patience = trial.suggest_int("patience", 20, 40)

    # DataLoaders (batch-size dependent)
    
    train_loader = DataLoader(
        df_train["graph"].tolist(),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=torch.cuda.is_available(), persistent_workers=True
    )
    val_loader = DataLoader(
        df_val["graph"].tolist(),
        batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=torch.cuda.is_available(), persistent_workers=True
    )
    test_loader = DataLoader(
        df_test["graph"].tolist(),
        batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=torch.cuda.is_available(), persistent_workers=True
    )

    # counts per split (for weights)
    n_train = counts_from_loader(train_loader, NUM_TASKS, device)
    n_val   = counts_from_loader(val_loader,   NUM_TASKS, device)
    n_test  = counts_from_loader(test_loader,  NUM_TASKS, device)

    # criteria (fixed ranges, split-specific counts)
    criterion_train = ContestWMAE(ranges_all, n_train, inverse_transform=inv_std).to(device)
    criterion_eval  = ContestWMAE(ranges_all, n_val,   inverse_transform=inv_std).to(device)
    criterion_test  = ContestWMAE(ranges_all, n_test,  inverse_transform=inv_std).to(device)

    # model
    sample_batch = next(iter(train_loader))
    in_channels = sample_batch.num_node_features
    edge_attr = getattr(sample_batch, "edge_attr", None)
    if edge_attr is None:
        raise RuntimeError("GINELayer expects edge attributes; got None.")
    edge_dim = edge_attr.size(-1) if edge_attr.dim() >= 2 else 1

    model = GINELayerUpgraded(
        in_channels=in_channels,
        edge_dim=edge_dim,
        hidden_channels=hidden,
        out_channels=NUM_TASKS,
        num_layers=num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train
    results = train_and_evaluate_edge(
        model=model,
        optimizer=optimizer,
        criterion_train=criterion_train,
        criterion_eval=criterion_eval,
        criterion_test=criterion_test,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs,
        early_stopping=True,
        patience=patience,
        # if your function accepts lr_schedulers or callbacks, you can add them here
    )

    # Get best validation wMAE (robust to different return shapes)
    if "best_val_loss" in results:
        best_val = float(results["best_val_loss"])
    elif "val_losses" in results and len(results["val_losses"]) > 0:
        best_val = float(min(results["val_losses"]))
    else:
        # as a fallback, evaluate on val preds if provided
        preds_z = results.get("val_preds")
        targets_z = results.get("val_targets")
        mask = results.get("val_mask")
        if preds_z is None or targets_z is None or mask is None:
            raise RuntimeError("Cannot derive validation metric from results.")
        with torch.no_grad():
            best_val = float(
                criterion_eval(
                    preds_z.to(device), targets_z.to(device), mask.float().to(device)
                ).item()
            )

    # Optional: report to Optuna for pruning (won't prune unless you use a pruner)
    trial.report(best_val, step=1)

    return best_val  # minimize wMAE

if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=30, show_progress_bar=True)
    print("\nBest value (val wMAE):", study.best_value)
    print("Best params:", study.best_params)

    # (Optional) Re-train with best params for longer epochs
    # You can load study.best_params and run a final fit with, say, epochs=200 and patience=40.
