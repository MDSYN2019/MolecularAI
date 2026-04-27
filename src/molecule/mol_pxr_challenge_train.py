import pandas as pd
import numpy as np
from datasets import load_dataset
from mol_functions import rdkit_globals, randomize_smiles, advanced_smiles_to_graph

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


train         = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TRAIN.csv")
test          = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TEST_BLINDED.csv")
train_counter  = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_counter-assay_TRAIN.csv")
test_structure = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_structure_TEST_BLINDED.csv")
train_single   = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_single_concentration_TRAIN.csv")


#splits = {'train': 'pxr-challenge_TRAIN.csv', 'test': 'pxr-challenge_TEST_BLINDED.csv'}
#df = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/" + splits["train"])

# We return the implementation for node_features, adjacencny matrix, edge_features and edge_indices
#advanced_smiles = [advanced_smiles_to_graph(smile) for smile in df['SMILES']]
#df['advanced_smiles'] = advanced_smiles
