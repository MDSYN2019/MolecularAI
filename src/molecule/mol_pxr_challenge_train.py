import pandas as pd
import numpy as np

from mol_functions import rdkit_globals, randomize_smiles, advanced_smiles_to_graph

splits = {'train': 'pxr-challenge_TRAIN.csv', 'test': 'pxr-challenge_TEST_BLINDED.csv'}
df = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/" + splits["train"])

# We return the implementation for node_features, adjacencny matrix, edge_features and edge_indices
advanced_smiles = [advanced_smiles_to_graph(smile) for smile in df['SMILES']]
df['advanced_smiles'] = advanced_smiles
