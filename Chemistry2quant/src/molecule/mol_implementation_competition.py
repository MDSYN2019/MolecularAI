 # -*- coding: utf-8 -*-
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from rdkit import Chem

from mol_functions import mol_to_graph, fix_tg_units, canon_polymer_smiles, load_official, load_tc_dataset, load_tg_dataset, load_ffv_dataset, resolve_conflicts, rdkit_globals



from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from mol_torch_gnn_implementation import (
    GINELayerUpgraded,
    train_and_evaluate_edge,  # cleaned version with align_targets + mask support
)

from sklearn.preprocessing import StandardScaler

# ---------------------------
# Config
# ---------------------------
PROPERTIES = ["Tg", "FFV", "Tc", "Density", "Rg"]

NUM_TASKS = len(PROPERTIES)
PATH = "/home/sang/Desktop/neurips-open-polymer-prediction-2025/molecule/"
TRAIN_CSV = f"{PATH}/train.csv"
TEST_CSV = f"{PATH}/test.csv"
SEED = 42
BATCH_SIZE = 64
HIDDEN = 384
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4

MINMAX_DICT = {
    "Tg": [-148.0297376, 472.25],
    "FFV": [0.2269924, 0.77709707],
    "Tc": [0.0465, 0.524],
    "Density": [0.748691234, 1.840998909],
    "Rg": [9.7283551, 34.672905605],
}

def add_weight_decay(model, weight_decay):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith('.bias') or 'norm' in n.lower() or 'bn' in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


def ranges_from_frame(df: pd.DataFrame, properties, device, p_low=1.0, p_high=99.0):
    rs = []
    for p in properties:
        s = pd.to_numeric(df[p], errors="coerce").dropna()
        if len(s) == 0:
            rs.append(1.0)  # fallback
            continue
        lo, hi = np.percentile(s, [p_low, p_high])
        rs.append(max(float(hi - lo), 1e-8))
    return torch.tensor(rs, dtype=torch.float32, device=device)


def ranges_from_minmax_dict(minmax_dict, device):
    mins = torch.tensor(
        [float(minmax_dict[p][0]) for p in PROPERTIES],
        dtype=torch.float32,
        device=device,
    )
    maxs = torch.tensor(
        [float(minmax_dict[p][1]) for p in PROPERTIES],
        dtype=torch.float32,
        device=device,
    )
    return (maxs - mins).clamp_min(1e-8)


def counts_from_loader(loader, device):
    cnt = torch.zeros(NUM_TASKS, dtype=torch.float32, device=device)
    for batch in loader:
        m = getattr(batch, "y_mask", None)
        if m is None:
            m = (~torch.isnan(batch.y)).float()
        if m.dim() == 1:
            m = m.view(-1, 1)
        cnt += m.sum(dim=0).to(device)
    return cnt.clamp_min(1.0)


def attach_u(df):
    df = df.copy()
    # ensure RangeIndex so index labels match positions if you ever need them
    df = df.reset_index(drop=True)
    # make sure the 'graph' column is object dtype (no numpy array weirdness)
    df["graph"] = df["graph"].astype(object)

    # 1) compute raw RDKit globals on TRAIN ONLY elsewhere, and fit scaler
    # u_train = np.stack([rdkit_globals(s) for s in df_train["SMILES"]], axis=0)
    # u_scaler = StandardScaler().fit(u_train)

    # 2) transform for this split and stash as tensors
    U_list = []
    for s in df["SMILES"]:
        v = rdkit_globals(s)                          # np.ndarray [U]
        v = u_scaler.transform(v.reshape(1, -1))[0]   # z-score by TRAIN stats
        U_list.append(torch.tensor(v, dtype=torch.float32))
    df["u"] = U_list

    # 3) mutate graphs in-place (no df.at reassign)
    for idx, g in df["graph"].items():               # items() gives true index label
        u_vec = df.at[idx, "u"].unsqueeze(0)         # [1, U]
        g.u = u_vec                                   # attach to existing Data

    return df


class ContestWMAE(nn.Module):
    """
    Contest wMAE (exact):
      loss = mean_over_batch( sum_i w_i * |pred_i - target_i| * mask_i )
    with w_i = (1 / r_i) * ( K / sqrt(n_i) ) / sum_j(1 / sqrt(n_j))

    Args
    ----
    ranges: tensor [T] with r_i (max - min) in ORIGINAL units
    counts: tensor [T] with n_i (label counts for this split)
    inverse_transform: optional callable mapping standardized -> original units
    """

    def __init__(
        self,
        ranges: torch.Tensor,
        counts: torch.Tensor,
        inverse_transform=None,
        eps: float = 1e-8,
    ):
        super().__init__()
        T = ranges.numel()
        inv_r = (1.0 / ranges.clamp_min(eps)).float()
        inv_sqrt_n = (1.0 / torch.sqrt(counts.clamp_min(1.0))).float()
        alpha = T / inv_sqrt_n.sum().clamp_min(eps)  # normalizes sum_j(...) to K=T
        w = inv_r * inv_sqrt_n * alpha  # [T]
        self.register_buffer("weights", w)
        self.inverse_transform = inverse_transform

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        # pred/target/mask: [B, T]
        if self.inverse_transform is not None:
            pred = self.inverse_transform(pred)
            target = self.inverse_transform(target)

        abs_err = (pred - target).abs() * mask  # [B, T]
        per_sample = (abs_err * self.weights).sum(dim=1)  # [B]
        return per_sample.mean()


# ---------------------------
# Utils
# ---------------------------


def split_df(df: pd.DataFrame, train=0.8, val=0.1, seed: int = SEED):
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(df), generator=gen)
    n = len(df)
    i_tr = idx[: int(train * n)]  # Generate training index
    i_va = idx[int(train * n) : int((train + val) * n)]  # Generate validation index
    i_te = idx[int((train + val) * n) :]  # Generate test index
    return df.iloc[i_tr].copy(), df.iloc[i_va].copy(), df.iloc[i_te].copy()


def compute_scalers(df_train: pd.DataFrame) -> dict:
    """ """
    scalers = {}
    for p in PROPERTIES:
        s = pd.to_numeric(df_train[p], errors="coerce")
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True)
        if pd.isna(sd) or sd == 0:
            sd = 1.0
        scalers[p] = (float(mu), float(sd))
    return scalers


def attach_multitask_targets(df: pd.DataFrame, scalers) -> pd.DataFrame:
    """ """
    df = df.copy()
    for i, row in df.iterrows():
        g: Data = row["graph"]
        y_vals, y_mask = [], []
        for p in PROPERTIES:
            val = row.get(p, None)
            if pd.notna(val):
                mu, sd = scalers[p]
                y_vals.append((float(val) - mu) / sd)  # z-score
                y_mask.append(1.0)
            else:
                y_vals.append(0.0)  # ignored by mask
                y_mask.append(0.0)
        g.y = torch.tensor(y_vals, dtype=torch.float)  # [T]
        g.y_mask = torch.tensor(y_mask, dtype=torch.float)  # [T]
        df.at[i, "graph"] = g
    return df


def make_loaders(df_train, df_val, df_test, batch_size=BATCH_SIZE):
    train_loader = DataLoader(
        df_train["graph"].tolist(), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        df_val["graph"].tolist(), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        df_test["graph"].tolist(), batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader, test_loader


# ---------------------------
# Data
# ---------------------------
supp_frames = []

official = load_official(TRAIN_CSV)

supp_frames.append(load_tc_dataset("train_supplement/dataset1.csv"))
supp_frames.append(load_tg_dataset("train_supplement/dataset3.csv"))
supp_frames.append(load_ffv_dataset("train_supplement/dataset4.csv", header_has_names=False))

big = pd.concat([official] + supp_frames, ignore_index=True, sort=False)
NUMERIC_COLS = ["Tg", "FFV", "Tc", "Density", "Rg"]  # adjust as needed
for c in NUMERIC_COLS:
    big[c] = pd.to_numeric(big[c], errors="coerce")
    
big["SMILES"] = big["SMILES"].map(canon_polymer_smiles)
big = big[big["SMILES"].notna()].copy()


if "FFV" in big.columns:
    big.loc[big["FFV"].notna(), "FFV"] = big["FFV"].clip(0.0, 1.0)
if "Density" in big.columns:
    big.loc[big["Density"].notna(), "Density"] = big["Density"].clip(lower=0.3, upper=3.0)

# optional: run fix_tg_units once more on the combined table
#merged = fix_tg_units(big, tg_col="Tg")
# median across duplicates (or plug your resolve_conflicts if you prefer)
agg = {c: "median" for c in NUMERIC_COLS}
big = (big
       .groupby("SMILES", as_index=False, sort=False)
       .agg(agg))

train_df = big  # now unique by SMILES

_ = pd.read_csv(TEST_CSV)  # kept if you later want to predict on test.csv

# Build PyG graphs for all rows

def randomize_smiles(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, doRandom=True, canonical=False) if m else s

train_df['SMILES'] = train_df['SMILES'].apply(randomize_smiles)
train_df["graph"] = train_df["SMILES"].apply(mol_to_graph)

u_train = np.stack([rdkit_globals(s) for s in train_df["SMILES"]], axis=0)
u_scaler = StandardScaler().fit(u_train)


# Split and scale (train-only stats)
df_train, df_val, df_test = split_df(train_df, train=0.8, val=0.1, seed=SEED)
scalers = compute_scalers(df_train)

# Attach multi-task targets + masks
df_train = attach_multitask_targets(df_train, scalers)
df_val = attach_multitask_targets(df_val, scalers)
df_test = attach_multitask_targets(df_test, scalers)


df_train = attach_u(df_train)
df_val   = attach_u(df_val)
df_test  = attach_u(df_test)

# DataLoaders
train_loader, val_loader, test_loader = make_loaders(
    df_train, df_val, df_test, batch_size=BATCH_SIZE
)
# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ranges_trainval = ranges_from_minmax_dict(MINMAX_DICT, device)  # r_i (original units)

ranges_train = ranges_from_frame(df_train, PROPERTIES, device)
ranges_val   = ranges_from_frame(df_val,   PROPERTIES, device)
ranges_test  = ranges_from_frame(df_test,  PROPERTIES, device)



n_train = counts_from_loader(train_loader, device)
n_val = counts_from_loader(val_loader, device)
n_test = counts_from_loader(test_loader, device)

mu = torch.tensor(
    [scalers[p][0] for p in PROPERTIES], dtype=torch.float32, device=device
)
sd = torch.tensor(
    [scalers[p][1] for p in PROPERTIES], dtype=torch.float32, device=device
)


def inv_std(z: torch.Tensor) -> torch.Tensor:
    return z * sd.to(z.device) + mu.to(z.device)

criterion_train = ContestWMAE(ranges_train, n_train, inverse_transform=inv_std).to(device)
criterion_eval  = ContestWMAE(ranges_val,   n_train,   inverse_transform=inv_std).to(device)
criterion_test  = ContestWMAE(ranges_test,  n_train,  inverse_transform=inv_std).to(device)

#criterion_train = ContestWMAE(ranges_trainval, n_train, inverse_transform=inv_std).to(
#    device
#)
#criterion_eval = ContestWMAE(ranges_trainval, n_val, inverse_transform=inv_std).to(
#    device
#)
#criterion_test = ContestWMAE(ranges_trainval, n_test, inverse_transform=inv_std).to(
#    device
#)


# Model
# ---------------------------
# Get feature sizes from one batch

sample = next(iter(train_loader))
in_channels = sample.num_node_features
edge_attr = getattr(sample, "edge_attr", None)
edge_dim    = sample.edge_attr.size(-1)
U = sample.u.size(1) if hasattr(sample, "u") else 0

if edge_attr is None:
    raise RuntimeError(
        "GINELayer expects edge features; got None. Supply edge_attr or switch to a non-edge GIN."
    )

edge_dim = edge_attr.size(-1) if edge_attr.dim() >= 2 else 1
#model = GINELayerUpgraded(
#    in_channels=in_channels,
#    edge_dim=edge_dim,
#    hidden_channels=HIDDEN,
#    out_channels=NUM_TASKS,  # multi-task
#    num_layers=4,
#    global_feat_dim=U,
#)

model = GINELayerUpgraded(
    in_channels=in_channels,
    edge_dim=edge_dim,
    out_channels=NUM_TASKS,
    hidden_channels=384,   # try 384 (or 320 if VRAM tight)
    num_layers=8,          # try 6–8
    dropout=0.25,           # 0.2–0.3 works well
    pooling="meanmax",     # <--- new
    use_gru=False,         # start false; add later if you want
    use_residual=True,
    norm="batch",          # BatchNorm in blocks; we add LayerNorm before head
    global_feat_dim=U,     # if you pass RDKit/global features
)
# ---------------------------
# Train
# ---------------------------

optimizer = torch.optim.AdamW(add_weight_decay(model, WEIGHT_DECAY), lr=LR, betas=(0.9,0.999), eps=1e-8)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5,
    patience=3,                 # ≈10 epochs given eval_every=5
    threshold_mode='abs', threshold=2e-4,
    cooldown=1, min_lr=1e-6, verbose=True
)


#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#    optimizer, mode='min', factor=0.5, patience=2, threshold=3e-4,
#    cooldown=3, min_lr=1e-6, verbose=True
#)


task_weights = torch.tensor([3.0, 1.0, 1.0, 1.0, 1.0])  # [Tg, FFV, Tc, Density, Rg]
# criterion = WeightedMaskedRMSE(task_weights=task_weights)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# mins_d, maxs_d = mins.to(device), maxs.to(device)
# criterion_train = MaskedWeightedScaledMAE(mins_d, maxs_d, train_w.to(device),
#                                          inverse_transform=inv_std if TRAIN_IN_STD else None)

# criterion_eval  = MaskedWeightedScaledMAE(mins_d, maxs_d, val_w.to(device),
#                                          inverse_transform=inv_std if TRAIN_IN_STD else None)

results = train_and_evaluate_edge(
    model=model,
    optimizer=optimizer,
    criterion_train=criterion_train,
    criterion_eval=criterion_eval,
    criterion_test=criterion_test,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    epochs=EPOCHS,
    early_stopping=True,
    eval_every=5,
    scheduler=scheduler,  # <--- pass it in
)

# save the model
torch.save(model.state_dict(), "model_best.pt")

print(
    "Done. Test loss:",
    results["test_losses"],
    "MAE:",
    results["test_mae"],
    "R2:",
    results["test_r2"],
)

"""They’re in z-score (standardized) space.

Your model’s outputs (test_preds) live in the same z-space (that’s what it was trained to predict).

test_targets are also in z-space, with missing labels filled with 0 as a placeholder but ignored via y_mask.

In contrast, your loss (ContestWMAE) converts preds/targets back to original units internally via inverse_transform before computing the weighted MAE. So:

results["test_losses"] ≈ contest wMAE in original units.

results["test_mae"] is a plain MAE in z-space (because it uses _masked_mae on the stored tensors).

"""


# Recomputing the original predictions and targets, then getting the mask as well

preds_z = results["test_preds"]  # [N, T] z-scores
targets_z = results["test_targets"]  # [N, T] z-scores (zeros where unlabeled)
mask = results["test_mask"].float()  # [N, T] 1 where labeled
# your train-based scalers (built earlier)
mu = torch.tensor([scalers[p][0] for p in PROPERTIES], dtype=torch.float32)
sd = torch.tensor([scalers[p][1] for p in PROPERTIES], dtype=torch.float32)

# invert to original units
preds_orig = preds_z * sd + mu  # broadcasting over T
targets_orig = targets_z * sd + mu

# per-property MAE in original units (ignoring missing)
per_prop_mae = ((preds_orig - targets_orig).abs() * mask).sum(0) / mask.sum(
    0
).clamp_min(1.0)

# if you want to recompute the official contest wMAE on these tensors:
contest_wmae = (
    criterion_test(  # uses inverse_transform internally, but it's fine to pass z
        preds_z.to(criterion_test.weights.device),
        targets_z.to(criterion_test.weights.device),
        mask.to(criterion_test.weights.device),
    ).item()
)


# Creating the properties for the test data

test_df = pd.read_csv(TEST_CSV)

test_graphs, test_ids = [], []


# building test graphs
for _, row in test_df.iterrows():
    s = str(row["SMILES"]).strip()
    try:
        g = mol_to_graph(s)  # g.u currently raw (unscaled) from graph_descriptors
        # Replace with TRAIN-scaled u:
        v = rdkit_globals(s)                                   # np [U]
        v = u_scaler.transform(v.reshape(1, -1))[0]            # z-score by TRAIN stats
        g.u = torch.tensor(v, dtype=torch.float32).unsqueeze(0)  # [1, U]

        test_graphs.append(g)
        test_ids.append(int(row["id"]))
    except Exception as e:
        print(f"[WARN] failed SMILES idx={row.name} :: {e}")

        
test_loader_only = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

# what is model eval doing here 
model.to(device); model.eval()
pred_chunks = []

with torch.no_grad():
    for data in test_loader_only:
        data = data.to(device)
        data.x = data.x.float()
        if getattr(data, "edge_attr", None) is not None:
            data.edge_attr = data.edge_attr.float()
        out = model(
            data.x, data.edge_index, data.edge_attr, batch=data.batch,
            u=getattr(data, "u", None),      # <<---- ADD THIS
        )
        out_orig = inv_std(out)
        pred_chunks.append(out_orig.detach().cpu())

pred_mat = torch.cat(pred_chunks, dim=0).numpy()
subm = pd.DataFrame(pred_mat, columns=PROPERTIES)
subm.insert(0, "id", test_ids)  # synthetic ids (0..N-1)
final_output = pd.merge(test_df, subm, on = 'id')
