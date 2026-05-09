from mol_functions import (
    rdkit_globals,
    advanced_smiles_to_graph,
    randomize_smiles,
)
from mol_multitask_utils import (
    split_df,
)
from mol_losses import StandardMAE
from mol_torch_gnn_implementation import  train_and_evaluate_edge
from mol_models import GINELayerUpgraded
from functools import partial

import optuna
import pandas as pd
import numpy as np

import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, CosineAnnealingLR

"""
OpenADMET Blind Challenge: PXR Induction Prediction — Developed graph-based molecular property prediction workflows using RDKit-derived descriptors, PyTorch Geometric molecular graphs, and assay-derived pEC50/Emax labels for a blinded ADMET benchmark task.
"""


BATCH_SIZE = 64
NUM_WORKERS = 4

def attach_targets(df: pd.DataFrame, target_col: str = "pEC50") -> pd.DataFrame:
    df = df.copy().reset_index(drop = True)
    df["graph"] = df["graph"].astype(object)

    for idx, graph in df["graph"].items():
        y_value = df.at[idx, target_col]
        graph.y = torch.tensor([y_value], dtype = torch.float32)

    return df
    
def advanced_smiles_to_pyg_data(smiles: str) -> Data:
    """
    Build a PyG Data object from ``advanced_smiles_to_graph`` outputs.
    """
    node_features, _, edge_features, edge_indices = advanced_smiles_to_graph(smiles)

    x = torch.tensor(node_features, dtype=torch.float32)
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    if len(edge_features) > 0:
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    else:
        edge_dim = 6  # 4 bond-type one-hots + is_conjugated + is_in_ring
        edge_attr = torch.empty((0, edge_dim), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


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
        graph.u = df.at[idx, "u"].unsqueeze(0) # attach the global features to the graph object in the dataframe, and unsqueeze to add a batch dimensoion
    return df


def objective(trial, sample_graph, train_loader, val_loader, test_loader, global_feat_dim: int, device: torch.device):
    """
    Objective function for Optuna hyperparameter optimization. This function will be called by optuna for each trial, and it should return a scalar value that represents the performance of the model with the given parameters. The goals of Optuna is to minimize this value.    
    """
    
    input_feature_dropout = trial.suggest_float("input_feature_dropout", 0.0, 0.2) # suggest a dropout rate for the input node features, between 0 and 0,2 
    edge_feature_dropout = trial.suggest_float("edge_feature_dropout", 0.0, 0.15)  # suggest a dropout rate for the input edge features, between 0 and 0.15 
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 192, 256, 384]) # suggest a hidden dimension for the GINELayerUpgraded, from the options 128, 256, and 512
    num_layers = trial.suggest_categorical("num_layers", [4, 6, 8]) # suggest a number of layers for the GINELayerUpgraded, between 3 and 10 
    dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3]) # suggest a dropout rate for the GINELayerUpGraded between 0 and 0.5

    
    lr = trial.suggest_float("lr", 1e-4, 5e-4, log=True) # suggest a learning rate for the AdawmW optimizer, between 1e-4 and 1e-2 on a log scale
    wd = trial.suggest_categorical("weight_decay",[1e-5, 1e-4, 3e-4]) # suggest a weight decay for the AdamW optmizer, from options 1e-5, 1e-4 and 3e-4

    scheduler_name = trial.suggest_categorical("scheduler", ["cosine", "onecycle"])# suggest a learning rate scheduler, from options "cosine" and "onecycle"

    
    model = GINELayerUpgraded(
        in_channels=sample_graph.x.shape[1],
        edge_dim=sample_graph.edge_attr.shape[1],
        out_channels=1,
        hidden_channels=hidden_dim, # use the hyperparameter for hidden dimensions
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


    # 3) optimizer + scheduler

    epochs = 80
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    if scheduler_name == "cosine":
        # cosine annealing
        scheduler = CosineAnnealingLR(optimizer, T_max = epochs, eta_min = 1e-6)

    else:
        scheduler = CosineAnnealingLR(optimizer, T_max = epochs, eta_min = 1e-6)

        
    criterion_train = StandardMAE()
    criterion_eval = StandardMAE()
    criterion_test = StandardMAE()
                                  
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
        device=device,
        eval_every=5,
        use_amp=True,
        scheduler = scheduler,
    )
    
    # return validation loss or -R² depending on your goal
    return min(results["val_losses"])


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
        loader_kwargs["prefetch_factor"] = 2#

    train_loader = DataLoader(df_train["graph"].tolist(), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(df_val["graph"].tolist(), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(df_test["graph"].tolist(), shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader

## Default config (primary assay)
#ds = load_dataset("openadmet/pxr-challenge-train-test")
#train = ds["train"]
#test  = ds["test"]
#
## Counter-assay config
#ds_counter = load_dataset("openadmet/pxr-challenge-train-test", "counter_assay")
#train_counter = ds_counter["train"]
#
## Structure config
#ds_structure = load_dataset("openadmet/pxr-challenge-train-test", "structure")
#test_structure = ds_structure["test"]
#
## Single-concentration config
#ds_single = load_dataset("openadmet/pxr-challenge-train-test", "single_concentration")
#train_single = ds_single["train"]
    
SEED = 42

train         = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TRAIN.csv")
test          = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TEST_BLINDED.csv")
train_counter  = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_counter-assay_TRAIN.csv")
test_structure = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_structure_TEST_BLINDED.csv")
train_single   = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_single_concentration_TRAIN.csv")


train["SMILES"] = train["SMILES"].apply(randomize_smiles) # randomize the SMILES strings to augment the data and prevent overfitting
train['graph'] = train['SMILES'].apply(advanced_smiles_to_pyg_data) # convert the SMILES strings to PyG Data objects and store them in a new column 'graph'

df_train, df_val, df_test = split_df(train, train=0.8, val=0.1, seed=SEED)

u_train = np.stack([rdkit_globals(s) for s in df_train["SMILES"]], axis=0)
u_scaler = StandardScaler().fit(u_train)
     
# attach 
df_train = attach_u_features(df_train, u_scaler) # attach the global features to the graph objects in the dataframe 
df_val = attach_u_features(df_val, u_scaler) # attach the global features to the graph objects in the dataframe 
df_test = attach_u_features(df_test, u_scaler)

# attach the target values to the graph objects in the dataframe, so that they can be easily accessed during training and evaluation
df_train = attach_targets(df_train)
df_val = attach_targets(df_val)
df_test = attach_targets(df_test)

train_loader, val_loader, test_loader = make_loaders(df_train, df_val, df_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_graph = df_train["graph"].iloc[0]
global_feat_dim = df_train["u"].iloc[0].shape[0]

#model = GINELayerUpgraded(
#    in_channels=sample_graph.x.shape[1],
#    edge_dim=sample_graph.edge_attr.shape[1],
#    out_channels=1,
#    hidden_channels=256,
#    num_layers=6,
#    dropout=0.2,
#    input_feature_dropout=0.1,
#    edge_feature_dropout=0.05,
#    pooling="attn",
#    use_gru=False,
#    use_residual=True,
#    norm="batch",
#    global_feat_dim=df_train["u"].iloc[0].shape[0],
#).to(device)
#
#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
#
#criterion_train = StandardMAE()
#criterion_eval = StandardMAE()
#criterion_test = StandardMAE()
#
#results = train_and_evaluate_edge(
#    model=model,
#    optimizer=optimizer,
#    criterion_train=criterion_train,
#    criterion_eval=criterion_eval,
#    criterion_test=criterion_test,
#    train_loader=train_loader,
#    val_loader=val_loader,
#    test_loader=test_loader,
#    epochs=120,
#    early_stopping=True,
#    device=device,
#    eval_every=5,
#    use_amp=True,
#)
#
#logging.info(
#    "Finished GINE training | test contest wMAE: %.5f | test MAE: %.5f | test R2: %.5f",
#    results["test_losses"],
#    results["test_mae"],
#    results["test_r2"],
#)
#
if __name__ == "__main__":
    objective_fn = partial(
        objective,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        sample_graph=sample_graph,
        global_feat_dim=global_feat_dim,
        device=device,
    )
    study = optuna.create_study(direction="minimize")  # or "maximize"
    study.optimize(objective_fn, n_trials=30)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
