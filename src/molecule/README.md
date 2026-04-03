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
   - `mol_mlops_template.py` (MLOps scaffold generator + simple run tracker)

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

## Improving training stability and test score (competition checklist)

If your target is better leaderboard/test performance in `mol_implementation_competition.py`,
use this order of operations:

1. **Fix evaluation quality first (before model changes)**
   - Use multiple seeds and average the validation metric. A single random split can be noisy.
   - Keep a strict holdout that is never used for hyperparameter decisions.
   - Track both overall wMAE and per-property MAE (Tg/FFV/Tc/Density/Rg), then optimize the worst property.

2. **Train longer with a safer schedule**
   - The current competition script uses a short run (`EPOCHS = 10`), which is often underfit for GINE stacks.
   - Increase epochs (e.g., 60-150) with early stopping and reduce-on-plateau LR scheduling.
   - Typical robust ranges to sweep:
     - LR: `3e-5` to `3e-4`
     - Weight decay: `1e-6` to `3e-4`
     - Dropout: `0.1` to `0.4`

3. **Tune capacity against overfitting**
   - Start from hidden size + depth in `GINELayerUpgraded`, then sweep:
     - `hidden_channels`: 256 / 384 / 512
     - `num_layers`: 6 / 8 / 10
   - If validation stalls early, reduce depth or increase dropout.
   - If train improves but val degrades, reduce capacity and/or increase regularization.

4. **Improve data signal before architecture changes**
   - Verify canonicalization + deduplication quality before graph conversion.
   - Check for label outliers and unit mismatches in supplemental datasets.
   - Consider lightweight SMILES augmentation (multiple randomizations) and average predictions.

5. **Use stronger validation strategy**
   - Prefer K-fold or repeated holdout to reduce variance in model selection.
   - Select checkpoints by validation wMAE, not just final epoch.

6. **Inference tricks that usually help**
   - Ensemble 3-5 independently seeded models.
   - Optionally use test-time augmentation (randomized SMILES) and average outputs.
   - Calibrate per-target bias using validation residuals (small additive correction per property).

7. **What to watch in logs**
   - `train ↓` and `val flat`: LR too low or under-capacity.
   - `train ↓` and `val ↑`: overfitting (increase dropout/weight decay, shorten patience).
   - One property dominates error: rebalance with loss weighting or targeted data cleaning for that property.

## If you feel lost

Start with step 1 and step 2 only. Once you can explain how a SMILES string becomes
`x`, `edge_index`, and `edge_attr`, the rest of the files become much easier to follow.

## New: MLOps template for polymer GNN experiments

You can scaffold a reproducible experiment layout for your polymer prediction workflow:

```bash
python src/molecule/mol_mlops_template.py --root-dir polymer_gnn_mlops
```

This creates:

- `configs/` with a base JSON config (`polymer_gnn.base.json`)
- `data/raw` and `data/processed` folders
- `artifacts/models` and `artifacts/reports`
- `runs/` for parameter + metric tracking

The `ExperimentTracker` class in `mol_mlops_template.py` gives a lightweight logging API
that you can call from your existing training scripts without introducing extra dependencies.
