import optuna
import pandas as pd
import numpy as np

from mol_multitask_utils import (
    split_df,
)

from mol_pxr_challenge_train import (objective,
                                     attach_targets,
                                     advanced_smiles_to_pyg_data,
                                     attach_u_features,
                                     make_loaders)

from mol_functions import (
    rdkit_globals,
    randomize_smiles,
)

from sklearn.preprocessing import StandardScaler
from mol_losses import StandardMAE
from mol_torch_gnn_implementation import  train_and_evaluate_edge
from mol_models import GINELayerUpgraded
from functools import partial


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


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=3)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="pxr_gine_tuning",
    )
    study.optimize(objective, n_trials=40, gc_after_trial=True)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
