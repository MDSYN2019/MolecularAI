import pandas as pd
import numpy as np
from datasets import load_dataset
from mol_functions import rdkit_globals, randomize_smiles, advanced_smiles_to_graph

import numpy as np
import pandas as pd
import torch
from torch import nn
import logging
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from mol_functions import (
    canon_polymer_smiles,
    load_ffv_dataset,
    load_official,
    load_tc_dataset,
    load_tg_dataset,
    rdkit_globals,
    randomize_smiles,
    advanced_smiles_to_graph,
)
from mol_losses import ContestWMAE, StandardMAE
from mol_multitask_utils import (
    attach_multitask_targets,
    compute_scalers,
    counts_from_loader,
    ranges_from_minmax_dict,
    split_df,
)
from mol_torch_gnn_implementation import  train_and_evaluate_edge
from mol_models import GINELayerUpgraded

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

model = GINELayerUpgraded(
    in_channels=sample_graph.x.shape[1],
    edge_dim=sample_graph.edge_attr.shape[1],
    out_channels=1,
    hidden_channels=256,
    num_layers=6,
    dropout=0.2,
    input_feature_dropout=0.1,
    edge_feature_dropout=0.05,
    pooling="attn",
    use_gru=False,
    use_residual=True,
    norm="batch",
    global_feat_dim=df_train["u"].iloc[0].shape[0],
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

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
    epochs=120,
    early_stopping=True,
    device=device,
    eval_every=5,
    use_amp=True,
)

logging.info(
    "Finished GINE training | test contest wMAE: %.5f | test MAE: %.5f | test R2: %.5f",
    results["test_losses"],
    results["test_mae"],
    results["test_r2"],
)

    
