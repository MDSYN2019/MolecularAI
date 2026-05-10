import optuna
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

from mol_functions import rdkit_globals

SEED = 42
N_BITS = 2048


def smiles_to_morgan_fp(smiles: str, radius: int = 2, n_bits: int = N_BITS) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    fps = np.stack([smiles_to_morgan_fp(s) for s in df["SMILES"]], axis=0)
    globals_arr = np.stack([rdkit_globals(s) for s in df["SMILES"]], axis=0).astype(np.float32)
    X = np.concatenate([fps, globals_arr], axis=1)
    y = df["Mean"].to_numpy(dtype=np.float32)
    return X, y


def build_model(trial: optuna.Trial):
    model_name = trial.suggest_categorical("model", ["hgb", "extra_trees", "random_forest"])

    if model_name == "hgb":
        return HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 255),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 50),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-8, 1e-1, log=True),
            max_bins=trial.suggest_int("max_bins", 64, 255),
            random_state=SEED,
        )

    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=trial.suggest_int("n_estimators", 400, 1600, step=200),
            max_depth=trial.suggest_int("max_depth", 8, 48),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.2, 0.4, 0.6]),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
            n_jobs=-1,
            random_state=SEED,
        )

    return RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 400, 1600, step=200),
        max_depth=trial.suggest_int("max_depth", 8, 48),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.2, 0.4, 0.6]),
        bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        n_jobs=-1,
        random_state=SEED,
    )


def objective(trial: optuna.Trial) -> float:
    model = build_model(trial)

    n_fp = N_BITS
    preproc = ColumnTransformer(
        transformers=[
            (
                "globals",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                slice(n_fp, None),
            ),
        ],
        remainder="passthrough",
        sparse_threshold=0.0,
    )

    pipe = Pipeline([
        ("preproc", preproc),
        ("model", model),
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_maes = []
    for train_idx, val_idx in cv.split(X_ALL):
        X_tr, X_va = X_ALL[train_idx], X_ALL[val_idx]
        y_tr, y_va = Y_ALL[train_idx], Y_ALL[val_idx]

        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_va)
        fold_maes.append(mean_absolute_error(y_va, preds))

    return float(np.mean(fold_maes))


if __name__ == "__main__":
    train = pd.read_csv("hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TRAIN.csv")
    X_ALL, Y_ALL = build_feature_matrix(train)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=3)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="pxr_classical_ml_tuning",
    )
    study.optimize(objective, n_trials=60, gc_after_trial=True)

    print("Best CV MAE:", study.best_value)
    print("Best params:", study.best_params)
