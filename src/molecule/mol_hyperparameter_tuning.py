from functools import partial

import optuna
import pandas as pd
import torch

from mol_functions import (
    mol_to_graph,
)

from mol_implementation_competition import compute_ml_dataset, property_to_compute
from mol_torch_gnn_implementation import GINELayer, train_and_evaluate_edge


"""
Defining the path for where to get the training and test data
"""

PATH = "/home/sang/Desktop/neurips-open-polymer-prediction-2025"
train_data = PATH + "/" + "train.csv"  # training data
test_data = PATH + "/" + "test.csv"  # test data
train_df = pd.read_csv(train_data)
test_df = pd.read_csv(test_data)

# Constructing the training dataset
train_df["graph"] = train_df["SMILES"].apply(lambda x: mol_to_graph(x))
train_df_graph = property_to_compute(train_df)
ml_dataset = compute_ml_dataset(train_df_graph)
sample_graph = ml_dataset["train_loader"].dataset[0]
in_channels = sample_graph.num_node_features
edge_dim = sample_graph.edge_attr.size(1)

# Define objective to accept ml_dataset
def objective(trial, ml_dataset, edge_dim=edge_dim):
    """
    Objective function to reduce ultimately.
    """
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 3, 10)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    wd = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)

    model = GINELayer(
        in_channels=43,
        edge_dim=edge_dim,
        hidden_channels=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        out_channels=1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    results = train_and_evaluate_edge(
        model,
        optimizer,
        torch.nn.MSELoss(),
        ml_dataset["train_loader"],
        ml_dataset["val_loader"],
        ml_dataset["test_loader"],
        epochs=200,
        early_stopping=True,
    )

    # return validation loss or -R² depending on your goal
    return results["val_losses"][-1]  # if minimizing loss
    # return -results["val_r2"]        # if maximizing R²


# Wrap with partial to fix ml_dataset
objective_fn = partial(objective, ml_dataset=ml_dataset, edge_dim=edge_dim)
study = optuna.create_study(direction="minimize")  # or "maximize"
study.optimize(objective_fn, n_trials=30)
print("Best params:", study.best_params)
