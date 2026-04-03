from mol_functions import mol_to_graph, fix_tg_units
from mol_torch_gnn_implementation import GINELayerUpgraded, train_and_evaluate_edge

from mol_losses import ContestWMAE
from mol_multitask_utils import (
    attach_multitask_targets,
    compute_scalers,
    counts_from_loader,
    ranges_from_minmax_dict,
    split_df,
)

from __future__ import annotations
import pandas as pd
import torch
import optuna
from torch_geometric.loader import DataLoader


def inv_std(z, mu=mu_t, sd=sd_t):
    return z * sd.to(z.device) + mu.to(z.device)

def set_seed(seed: int = SEED) -> None:
    import numpy as np
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_loader(graphs, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    use_cuda = torch.cuda.is_available()
    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )

def extract_best_val(results: dict, criterion_eval: ContestWMAE, device: torch.device) -> float:
    if "best_val_loss" in results:
        return float(results["best_val_loss"])
    if "val_losses" in results and len(results["val_losses"]) > 0:
        return float(min(results["val_losses"]))

    preds_z = results.get("val_preds")
    targets_z = results.get("val_targets")
    mask = results.get("val_mask")
    if preds_z is None or targets_z is None or mask is None:
        raise RuntimeError("Cannot derive validation metric from results.")
    with torch.no_grad():
        return float(
            criterion_eval(
                preds_z.to(device), targets_z.to(device), mask.float().to(device)
            ).item()
        )

    
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

# ---------- DATA (built once) ----------
set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_df = pd.read_csv(TRAIN_CSV)
train_df = fix_tg_units(train_df)
train_df["graph"] = train_df["SMILES"].apply(mol_to_graph) # this may need to be changed

df_train, df_val, df_test = split_df(train_df, train=0.8, val=0.1, seed=SEED) # split to train, validation and test datasets
scalers = compute_scalers(df_train, PROPERTIES)

df_train = attach_multitask_targets(df_train, scalers, PROPERTIES) 
df_val = attach_multitask_targets(df_val, scalers, PROPERTIES)
df_test = attach_multitask_targets(df_test, scalers, PROPERTIES)

mu_t = torch.tensor([scalers[p][0] for p in PROPERTIES], dtype=torch.float32, device=device)
sd_t = torch.tensor([scalers[p][1] for p in PROPERTIES], dtype=torch.float32, device=device)


ranges_all = ranges_from_minmax_dict(MINMAX_DICT, PROPERTIES, device)



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
    train_loader = build_loader(df_train["graph"].tolist(), batch_size, shuffle=True, num_workers=4)
    val_loader = build_loader(df_val["graph"].tolist(), batch_size, shuffle=False, num_workers=2)
    test_loader = build_loader(df_test["graph"].tolist(), batch_size, shuffle=False, num_workers=2)

    # counts per split (for weights)
    n_train = counts_from_loader(train_loader, NUM_TASKS, device)
    n_val = counts_from_loader(val_loader, NUM_TASKS, device)
    n_test = counts_from_loader(test_loader, NUM_TASKS, device)

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
    best_val = extract_best_val(results, criterion_eval, device)
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
