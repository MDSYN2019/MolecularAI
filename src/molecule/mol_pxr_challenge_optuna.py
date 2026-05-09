import optuna
import pandas as pd
import numpy as np
import torch
from functools import partial

from mol_multitask_utils import split_df
from mol_pxr_challenge_train import (
    objective,
    attach_targets,
    advanced_smiles_to_pyg_data,
    attach_u_features,
    make_loaders,
)
from mol_functions import rdkit_globals, randomize_smiles
from sklearn.preprocessing import StandardScaler

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


if __name__ == "__main__":
    sample_graph = df_train["graph"].iloc[0]
    global_feat_dim = df_train["u"].iloc[0].shape[0]

    bound_objective = partial(
        objective,
        sample_graph=sample_graph,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        global_feat_dim=global_feat_dim,
        device=device,
    )

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
