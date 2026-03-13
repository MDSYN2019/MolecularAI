# PyTorch Geometric learning path (molecules)

This folder contains the PyTorch Geometric (PyG) work. The code evolved over time, so this guide
splits it into a clear, step-by-step learning path. Follow the **recommended order** to build
intuition from data prep → graph construction → modeling → training/tuning → experiments.

## Recommended learning order

1. **Understand molecule inputs and atom/bond features**
   - **Start here:** `mol_functions_chem.py`
   - Focus on the atom/bond featurization helpers and the feature definitions.
   - Goal: know what information gets encoded into node/edge features.

2. **Create PyG graphs from SMILES**
   - **Read next:** `mol_torch_gnn_implementation.py` (see `smiles_to_pytorch_graph`)
   - Focus on how RDKit molecules are converted into PyG `Data` objects.
   - Goal: understand node features, edge indices, and edge attributes.

3. **Train a baseline GNN**
   - **Continue in:** `mol_torch_gnn_implementation.py`
   - Look at the GNN layer definitions and the training loop.
   - Goal: run a baseline model end-to-end and interpret metrics.

4. **Tune hyperparameters**
   - **Then:** `mol_hyperparameter_tuning.py`
   - **Advanced:** `mol_hyperparameter_tuning_GINEUpgraded.py`
   - Goal: learn how learning rate, weight decay, hidden sizes, and message passing depth
     affect performance.

5. **Production-style evaluation / competition workflow**
   - **Later:** `mol_implementation_competition.py`
   - Goal: see how predictions are generated in chunks for larger datasets.

6. **Reference / experiments / troubleshooting**
   - `mol_backup.py` (older experiments, alternative layers)
   - `mol_troubleshooting.py` (debug notes and utility snippets)
   - `mol_feature_store.py`, `mol_features.py`, `mol_functions.py` (feature experiments)
   - `mol_xgboost_model.py` (non-GNN baseline)

## Concept map (what each file teaches)

| Step | Concept | File(s) |
| --- | --- | --- |
| 1 | Atom/bond feature engineering | `mol_functions_chem.py` |
| 2 | RDKit → PyG graph conversion | `mol_torch_gnn_implementation.py` |
| 3 | GNN model + training loop | `mol_torch_gnn_implementation.py` |
| 4 | Hyperparameter search | `mol_hyperparameter_tuning.py`, `mol_hyperparameter_tuning_GINEUpgraded.py` |
| 5 | Inference workflow | `mol_implementation_competition.py` |
| 6 | Experiments / backups | `mol_backup.py`, `mol_troubleshooting.py` |

## Suggested workflow (practical)

1. Read the feature helpers and scan the feature list in `mol_functions_chem.py`.
2. Walk through `smiles_to_pytorch_graph` and build a single `Data` object.
3. Train the baseline in `mol_torch_gnn_implementation.py` on a small subset.
4. Add/modify a feature (atom or bond) and re-train to see impact.
5. Tune the model in `mol_hyperparameter_tuning.py`.
6. Compare with the upgraded GINE workflow.

## If you feel lost

Start with step 1 and step 2 only. Once you can explain how a SMILES string becomes
`x`, `edge_index`, and `edge_attr`, the rest of the files become much easier to follow.
