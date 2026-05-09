import optuna
import pandas as pd
import numpy as np
import torch

from mol_multitask_utils import split_df
from mol_pxr_challenge_train import (

    attach_targets,
    advanced_smiles_to_pyg_data,
    attach_u_features,
    make_loaders,
)
from mol_functions import rdkit_globals, randomize_smiles
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from mol_models import GINELayerUpgraded
from mol_losses import StandardMAE
from mol_torch_gnn_implementation import train_and_evaluate_edge

SEED = 42


train = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TRAIN.csv")

# Augment canonical strings with randomized equivalents before graph conversion.
train["SMILES"] = train["SMILES"].apply(randomize_smiles)
train["graph"] = train["SMILES"].apply(advanced_smiles_to_pyg_data)

df_train, df_val, df_test = split_df(train, train=0.8, val=0.1, seed=SEED)

u_train = np.stack([rdkit_globals(s) for s in df_train["SMILES"]], axis=0)
u_scaler = StandardScaler().fit(u_train)

df_train = attach_u_features(df_train, u_scaler)
df_val = attach_u_features(df_val, u_scaler)
df_test = attach_u_features(df_test, u_scaler)

df_train = attach_targets(df_train)
df_val = attach_targets(df_val)
df_test = attach_targets(df_test)

train_loader, val_loader, test_loader = make_loaders(df_train, df_val, df_test)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_graph = df_train["graph"].iloc[0]
global_feat_dim = df_train["u"].iloc[0].shape[0]

def objective(trial: optuna.Trial) -> float:
    input_feature_dropout = trial.suggest_float("input_feature_dropout", 0.0, 0.2)
    edge_feature_dropout = trial.suggest_float("edge_feature_dropout", 0.0, 0.15)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 192, 256, 384])
    num_layers = trial.suggest_categorical("num_layers", [4, 6, 8])
    dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3])
    lr = trial.suggest_float("lr", 1e-4, 5e-4, log=True)
    weight_decay = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 3e-4])

    model = GINELayerUpgraded(
        in_channels=sample_graph.x.shape[1],
        edge_dim=sample_graph.edge_attr.shape[1],
        out_channels=1,
        hidden_channels=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        input_feature_dropout=input_feature_dropout,
        edge_feature_dropout=edge_feature_dropout,
        pooling="attn",
        use_gru=True,
        use_residual=True,
        norm="batch",
        global_feat_dim=global_feat_dim,
    ).to(device)

    epochs = 80
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    results = train_and_evaluate_edge(
        model=model,
        optimizer=optimizer,
        criterion_train=StandardMAE(),
        criterion_eval=StandardMAE(),
        criterion_test=StandardMAE(),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs,
        early_stopping=True,
        device=device,
        eval_every=5,
        use_amp=True,
        scheduler=scheduler,
    )
    return min(results["val_losses"])


if __name__ == "__main__":

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=3)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="pxr_gine_tuning",
    )
    study.optimize(bound_objective, n_trials=40, gc_after_trial=True)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
