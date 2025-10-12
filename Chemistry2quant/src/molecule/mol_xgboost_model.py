from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import train_test_split
from scipy import sparse

# --- config ---
PROPERTIES = ["Tg", "FFV", "Tc", "Density", "Rg"]
N_BITS = 2048
RADIUS = 2
SEED = 42


def morgan_bits(smi, n_bits=N_BITS, radius=RADIUS):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        arr = np.zeros(n_bits, dtype=np.uint8)
        return arr
    bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def scaffold_smiles(s) -> str:
    """Return Bemis–Murcko scaffold SMILES for a SMILES string s.
       Safe on NaN/invalid; returns '' on failure."""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if not s:
        return ""
    m = Chem.MolFromSmiles(s)
    if m is None:
        return ""
    try:
        # IMPORTANT: pass by keyword, not positional
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        return ""

def scaffold_split(df, seed=42, train_frac=0.8, val_frac=0.1):
    # make sure SMILES are strings; avoid NA issues
    smiles = df["SMILES"].astype(str).fillna("")
    scaf = smiles.map(scaffold_smiles)
    df2 = df.copy()
    df2["__scaffold__"] = scaf

    scafs = df2["__scaffold__"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(scafs)

    n = len(scafs)
    n_tr = int(train_frac * n)
    n_va = int((train_frac + val_frac) * n)

    set_tr = set(scafs[:n_tr])
    set_va = set(scafs[n_tr:n_va])

    tr = df2[df2["__scaffold__"].isin(set_tr)].drop(columns="__scaffold__")
    va = df2[df2["__scaffold__"].isin(set_va)].drop(columns="__scaffold__")
    te = df2[~df2.index.isin(tr.index.union(va.index))].drop(columns="__scaffold__")
    return tr, va, te


def rdkit_descriptors(m):
    # a compact set that usually helps:
    return np.array([
        Descriptors.MolWt(m),
        Descriptors.MolLogP(m),
        rdMolDescriptors.CalcTPSA(m),
        rdMolDescriptors.CalcNumHBD(m),
        rdMolDescriptors.CalcNumHBA(m),
        rdMolDescriptors.CalcNumRotatableBonds(m),
        rdMolDescriptors.CalcNumAromaticRings(m),
        rdMolDescriptors.CalcFractionCSP3(m),
        m.GetNumHeavyAtoms(),
        sum(1 for a in m.GetAtoms() if a.GetSymbol()=='F'),
        sum(1 for a in m.GetAtoms() if a.GetAtomicNum()==0),  # wildcard '*'
    ], dtype=np.float32)

DESC_DIM = 11  # keep in sync with rdkit_descriptors above

def featurize_smiles(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return np.hstack([np.zeros(N_BITS, np.uint8), np.zeros(DESC_DIM, np.float32)])
    return np.hstack([morgan_bits(smi), rdkit_descriptors(m)])

def build_X(df):
    # returns scipy CSR (sparse) – good for XGB hist
    X_rows = []
    for s in df["SMILES"].astype(str).values:
        X_rows.append(featurize_smiles(s))
    X = np.vstack(X_rows)
    # bits are {0,1}, descs are float – no need to scale for trees
    return sparse.csr_matrix(X)  # [N, 2048+11]

# official wMAE (same formula you used in PyTorch)
def contest_wmae(y_true_matrix, y_pred_matrix, mask_matrix, ranges, counts):
    # y_* shape: [N, T] in ORIGINAL units
    abs_err = np.abs(y_pred_matrix - y_true_matrix) * mask_matrix
    # MAE per task over labeled rows
    denom = mask_matrix.sum(axis=0).clip(min=1.0)
    mae_t = abs_err.sum(axis=0) / denom  # [T]
    # weights
    T = len(ranges)
    inv_r = 1.0 / np.clip(ranges, 1e-8, None)
    inv_sqrt_n = 1.0 / np.sqrt(np.clip(counts, 1.0, None))
    alpha = T / inv_sqrt_n.sum()
    w = inv_r * inv_sqrt_n * alpha
    return float((w * mae_t).sum()), mae_t, w

def compute_wmae_valid(y_true_df, y_pred_df, ranges_train,  # dict or Series in ORIGINAL units
                       K=None):
    present = [t for t in PROPERTIES if y_true_df[t].notna().sum() > 0]
    if not present:
        return {t: np.nan for t in PROPERTIES}, np.nan, {}

    # per-task MAE on rows that have labels
    per_mae = {}
    n_val = {}
    for t in present:
        m = y_true_df[t].notna()
        per_mae[t] = np.mean(np.abs(y_true_df.loc[m, t] - y_pred_df.loc[m, t]))
        n_val[t] = int(m.sum())

    # contest weights: w_i ∝ (1 / range_i) * (K / sqrt(n_i))
    if K is None:
        K = len(present)  # same normalization you used before
    inv_r = {t: 1.0 / max(float(ranges_train[t]), 1e-8) for t in present}
    base = {t: inv_r[t] * (K / np.sqrt(max(n_val[t], 1))) for t in present}
    Z = sum(base.values())
    w = {t: base[t] / Z for t in present}

    # scaled MAE and weighted mean
    scaled_mae = {t: per_mae[t] * inv_r[t] for t in present}
    wmae = sum(w[t] * scaled_mae[t] for t in present)

    # fill NaN for absent tasks for reporting
    for t in PROPERTIES:
        per_mae.setdefault(t, np.nan)
        w.setdefault(t, 0.0)

    return per_mae, float(wmae), w

# -------------------------
# load & prepare
# -------------------------
PATH = "/home/sang/Desktop/neurips-open-polymer-prediction-2025/molecule/"
train_df = pd.read_csv(f"{PATH}/train.csv")
# keep your Kelvin->C fix for Tg
from mol_functions import fix_tg_units
train_df = fix_tg_units(train_df)

# scaffold split
df_tr, df_va, df_te = scaffold_split(train_df, seed=SEED)

# features
X_tr = build_X(df_tr)
X_va = build_X(df_va)

# label availability (counts per task on TRAIN for weights)
counts = []
ranges = []
for p in PROPERTIES:
    s = pd.to_numeric(df_tr[p], errors="coerce")
    counts.append(int(s.notna().sum()))
    lo, hi = np.nanpercentile(s, [1.0, 99.0])
    ranges.append(max(hi - lo, 1e-8))
counts = np.array(counts, dtype=float)
ranges = np.array(ranges, dtype=float)

# -------------------------
# train one XGB per task
# -------------------------
models = {}
val_preds = []
val_targets = []
val_masks = []

for t, prop in enumerate(PROPERTIES):
    y_tr = pd.to_numeric(df_tr[prop], errors="coerce")
    y_va = pd.to_numeric(df_va[prop], errors="coerce")

    tr_mask = y_tr.notna().values
    va_mask = y_va.notna().values

    dtr_X = X_tr[tr_mask]
    dtr_y = y_tr.values[tr_mask].astype(np.float32)

    dva_X = X_va[va_mask]
    dva_y = y_va.values[va_mask].astype(np.float32)

    # A solid starting config (fast + accurate). Tweak if needed.
    model = xgb.XGBRegressor(
        n_estimators=5000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_alpha=1e-3,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=SEED,
        objective="reg:squarederror",  # you can try "reg:absoluteerror"
        n_jobs=-1,
        eval_metric="mae",
    )
    model.fit(
        dtr_X,
        dtr_y,
        eval_set=[(dva_X, dva_y)],
        verbose=False,
    )

    models[prop] = model
    # collect val preds/targets with mask for wMAE
    # (fill the full-length val vector with NaN, write preds at labeled indices)
    prop_pred = np.full(len(df_va), np.nan, dtype=np.float32)
    prop_pred[va_mask] = model.predict(dva_X)

    val_preds.append(prop_pred)
    val_targets.append(y_va.values.astype(np.float32))
    val_masks.append(va_mask.astype(np.float32))

# stack [T, N] -> [N, T]
val_preds = np.vstack(val_preds).T
val_targets = np.vstack(val_targets).T
val_masks = np.vstack(val_masks).T

# compute official wMAE on validation
wmae, per_task_mae, weights = contest_wmae(val_targets, val_preds, ranges)

print("Validation per-task MAE:", dict(zip(PROPERTIES, per_task_mae)))
print("Validation wMAE:", wmae)
print("Task weights used:", dict(zip(PROPERTIES, weights)))
