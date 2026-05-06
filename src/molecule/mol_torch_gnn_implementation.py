
"""
GNNs are specific layers that input a graph and output a graph.
For each atom(node), we need to define a feature vector.
For example:

Atom type.
Formal charge.
Hybridization State.
Aromaticity.
Number of connected hydrogens.

=> Each row represents one atom in the molecule, with it's feature encoded

"""

from mol_functions import (
    advanced_smiles_to_graph,
    mol_to_graph,
    align_targets,
)

import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm as tqdm

from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet

from torch_geometric.nn import (
    GCNConv,
    GINConv,
    GINEConv,
    GlobalAttention,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)

logging.basicConfig(level=logging.INFO)

from mol_models import GCNLayer

class MaskedWeightedScaledMAE(nn.Module):
    """
    Matches the Open Polymer wMAE:
      - per-task MAE with missing-target mask
      - min–max scaling per task
      - weighted average over tasks

    Args
    ----
    mins, maxs: 1D tensors [T] of per-property min / max in ORIGINAL UNITS.

    prop_weights: 1D tensor [T] that sums to 1 (e.g., from label availability).

    inverse_transform: optional callable(pred)->pred_in_original_units and
                       callable(target)->target_in_original_units (affine recommended).
                       If None, assumes inputs already in original units.


    eps: small constant to avoid division by zero in ranges.

    reduction: "mean" returns scalar; "none" returns vector of scaled MAE per task.
    """

    def __init__(
        self,
        mins: torch.Tensor,
        maxs: torch.Tensor,
        prop_weights: torch.Tensor,
        inverse_transform=None,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super().__init__()
        assert reduction in {"mean", "none"}

        self.register_buffer("mins", mins.float()) # what is register_buffer doing here? - we keep track of the minimum values
        self.register_buffer("maxs", maxs.float()) # we keep track of the maximum vlaues

        w = prop_weights.float() # convert to floats
        self.register_buffer("weights", w / w.sum().clamp_min(1e-12)) # compute and register the weights  - normalize
        self.inverse_transform = inverse_transform # inverse transform
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        # Align shapes upstream if needed: pred,target,mask -> [B, T]
        if self.inverse_transform is not None:
            # Map back to ORIGINAL UNITS before computing MAE
            pred = self.inverse_transform(pred)  # convert prediction to original units
            target = self.inverse_transform(target) # convert target to original units

        # per-sample absolute error, masked
        abs_err = (pred - target).abs() * mask  # [B, T]

        # MAE per task over rows with labels
        denom_t = mask.sum(dim=0).clamp_min(1.0)  # [T]
        mae_t = abs_err.sum(dim=0) / denom_t  # [T]

        # min–max scale per task
        rng = (self.maxs - self.mins).clamp_min(self.eps)  # [T]
        scaled_mae_t = mae_t / rng  # [T]

        if self.reduction == "none":
            return scaled_mae_t

        # Weighted average across tasks -> scalar
        return (self.weights * scaled_mae_t).sum()


class MaskedRMSE(nn.Module):
    def __init__(self, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        assert reduction in {"mean", "none"}
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        # pred/target/mask: [B, T]
        # If a row has no valid targets (mask==0), we drop it from the batch average.
        mse = ((pred - target) ** 2) * mask
        per_row_valid = mask.sum(dim=1)  # [B]
        per_row_mse = mse.sum(dim=1) / per_row_valid.clamp_min(1.0)  # [B]
        per_row_rmse = torch.sqrt(per_row_mse + self.eps)  # [B]

        if self.reduction == "none":
            return per_row_rmse
        # average only over rows that had at least one valid target
        denom = (per_row_valid > 0).float().sum().clamp_min(1.0)
        return per_row_rmse.sum() / denom


class RMSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(RMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, input, target):
        return torch.sqrt(self.mse_loss(input, target))

def mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.mean(torch.abs(preds - targets)).item()

def r2_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")




def train_and_evaluate(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    test_loader,
    epochs=50,
    early_stopping=True,
):
    train_losses = []
    val_losses = []
    float("inf")

    for epoch in tqdm.tqdm(range(1, epochs + 1)):
        # training mode
        model.train()
        train_loss = 0
        for data in train_loader:
            optimizer.zero_grad()  # Zero the gradient
            out = model(data.x.float(), data.edge_index, data.batch)
            loss = criterion(out, data.y)  # Compute the difference in the y and y_pred
            loss.backward()  # Compute the backward propagation to compute the Delta to subtract from the weights
            optimizer.step()
            train_loss += loss.item() * data.num_graphs

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation - change to model validation mode
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                out = model(data.x.float(), data.edge_index, data.batch)
                loss = criterion(out, data.y)
                val_loss += loss.item() * data.num_graphs

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if epoch % 5 == 0:
            logging.info(f"Epoch {epoch}: Train Loss = {train_loss:}, Val loss {val_loss}")

    model.eval()
    test_loss = 0
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for data in test_loader:
            out = model(data.x.float(), data.edge_index, data.batch)
            loss = criterion(out, data.y)
            test_loss += loss.item() * data.num_graphs
            # What is out.cpu() doing?
            test_preds.append(out.cpu())
            test_targets.append(data.y.cpu())

    test_loss /= len(test_loader.dataset)
    test_preds = torch.cat(test_preds, dim=0)
    test_targets = torch.cat(test_targets, dim=0)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_losses": test_loss,
        "test_preds": test_preds,
        "test_targets": test_targets,
        "model": model,
    }

# Train models with different pooling methods
def compare_pooling_methods(dataset, train_loader, val_loader, test_loader):
    # Get input dimension
    sample = dataset[0]
    in_channels = sample.x.shape[1]
    # Common parameters
    hidden_channels = 64
    out_channels = 1  # Regression task
    lr = 0.001
    weight_decay = 5e-4
    epochs = 50
    criterion = nn.L1Loss()
    # Initialize models with different pooling methods
    models = {
        "Mean Pooling": GCNLayer(in_channels, hidden_channels, out_channels),
        "Max Pooling": GCNLayer(in_channels, hidden_channels, out_channels),
        "Sum Pooling": GCNLayer(in_channels, hidden_channels, out_channels),
    }

    # Train and evaluate each model
    results = {}

    for name, model in models.items():
        logging.info(f"\nTraining model with {name}...")
        optimizer = torch.optim.Adam(  # Need to look up adam
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # go through the train_loader, train the model.
        # then use the validation dataset to validate the model
        # then test the performance on the test data
        results[name] = train_and_evaluate(
            model, optimizer, criterion, train_loader, val_loader, test_loader, epochs
        )

    return results


def drop_edges_(data, p: float):
    if p <= 0 or getattr(data, 'edge_index', None) is None:
        return data
    E = data.edge_index.size(1)
    keep = torch.rand(E, device=data.edge_index.device) > p
    data.edge_index = data.edge_index[:, keep]
    if getattr(data, 'edge_attr', None) is not None:
        data.edge_attr = data.edge_attr[keep]
    return data

def train_and_evaluate_edge(
    model,
    optimizer,
    criterion_train,
    criterion_eval,
    criterion_test,
    train_loader,
    val_loader,
    test_loader,
    epochs=200,
    early_stopping=True,
    device=None,
    eval_every=5,   # validate every N epochs
    use_amp=True,   # mixed precision on CUDA
    scheduler=None, # <--- NEW: pass a ReduceLROnPlateau here
):
    """
    Faster train+eval for (edge-aware) GNNs.

    - Multi-task with optional masks (data.y_mask or NaN->mask)
    - Shape alignment via align_targets(out, y, y_mask)
    - Contest-aligned losses: pass split-specific criteria

    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    amp_enabled = use_amp and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    def forward_model(data, training: bool):
        if training:
            data = data.clone()
            data = drop_edges_(data, p=0.15)

        # if we do have edges in our input
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            return model(
                data.x.float(),
                data.edge_index,
                data.edge_attr.float(),
                batch=data.batch,
                u=getattr(data, "u", None),   # <<---- ADD THIS
            )
        else:
            return model(
                data.x.float(),
                data.edge_index,
                batch=data.batch,
                u=getattr(data, "u", None),   # <<---- AND HERE (if you ever use no edge_attr)
            )

    def _masked_flat(pred, target, mask):
        if mask is None:
            return pred.reshape(-1), target.reshape(-1)
        m = mask > 0
        return pred[m], target[m]

    def _masked_mae(pred, target, mask):
        p, t = _masked_flat(pred, target, mask)
        if p.numel() == 0:
            return float("nan")
        return torch.mean(torch.abs(p - t)).item()

    def _masked_r2(pred, target, mask):
        p, t = _masked_flat(pred, target, mask)
        if p.numel() == 0:
            return float("nan")
        ss_res = torch.sum((t - p) ** 2)
        ss_tot = torch.sum((t - torch.mean(t)) ** 2)
        return (1 - ss_res / ss_tot).item() if ss_tot != 0 else float("nan")

    train_losses, val_losses = [], []
    best_val, best_state = float("inf"), None
    patience, patience_counter = 10, 0

    for epoch in range(1, epochs + 1):
        # ---------- Train ----------
        model.train()
        t0 = time.time()
        running, train_graphs = 0.0, 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                out = forward_model(data, training=True)
                y_raw, y_mask = data.y, getattr(data, "y_mask", None)
                if y_mask is None:  # NaN -> mask
                    y_mask = (~torch.isnan(y_raw)).float()
                    y_raw = torch.nan_to_num(y_raw, nan=0.0)
                y, y_mask = align_targets(out, y_raw, y_mask)
                loss = criterion_train(out, y, y_mask)

            scaler.scale(loss).backward()
            # gradient clipping part
            scaler.unscale_(optimizer)  # required before clipping when using AMP
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * data.num_graphs
            train_graphs += data.num_graphs

        train_loss = running / max(1, train_graphs)
        train_losses.append(train_loss)
        t1 = time.time()
        train_speed = train_graphs / (t1 - t0 + 1e-6)

        # ---------- Validation (every eval_every epochs) ----------
        do_val = (epoch % eval_every == 0) or (epoch == epochs)
        if do_val:
            model.eval()
            running, val_graphs = 0.0, 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                        out = forward_model(data, training=False)
                        y_raw, y_mask = data.y, getattr(data, "y_mask", None)
                        if y_mask is None:
                            y_mask = (~torch.isnan(y_raw)).float()
                            y_raw = torch.nan_to_num(y_raw, nan=0.0)
                        y, y_mask = align_targets(out, y_raw, y_mask)
                        loss = criterion_eval(out, y, y_mask)
                    running += loss.item() * data.num_graphs
                    val_graphs += data.num_graphs
            val_loss = running / max(1, val_graphs)
            val_losses.append(val_loss)

            # ---- NEW: step the scheduler on the validation metric ----
            if scheduler is not None:
                scheduler.step(val_loss)
            curr_lr = optimizer.param_groups[0]["lr"]

            logging.info(
                f"Epoch {epoch:04d} | Train {train_loss:.4f} "
                f"({train_speed:.1f} graphs/s) | Val {val_loss:.4f} | LR {curr_lr:.2e}"
            )

            if early_stopping:
                if val_loss + 1e-9 < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(f"Early stopping at epoch {epoch}. Best Val {best_val:.4f}")
                        break
        else:
            logging.info(f"Epoch {epoch:04d} | Train {train_loss:.4f} ({train_speed:.1f} graphs/s)")

    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    # ---------- Test ----------
    model.eval()
    test_running, test_graphs = 0.0, 0
    preds, targets, masks = [], [], []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                out = forward_model(data, training=False)
                y_raw, y_mask = data.y, getattr(data, "y_mask", None)
                if y_mask is None:
                    y_mask = (~torch.isnan(y_raw)).float()
                    y_raw = torch.nan_to_num(y_raw, nan=0.0)
                y, y_mask = align_targets(out, y_raw, y_mask)
                loss = criterion_test(out, y, y_mask)
            test_running += loss.item() * data.num_graphs
            test_graphs += data.num_graphs
            preds.append(out.cpu())
            targets.append(y.cpu())
            masks.append(None if y_mask is None else y_mask.cpu())

    test_loss = test_running / max(1, test_graphs)  # contest wMAE on test
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    ymask = (
        None if all(m is None for m in masks)
        else torch.cat([m if m is not None else torch.ones_like(preds) for m in masks], dim=0)
    )
    test_mae = _masked_mae(preds, targets, ymask)
    test_r2  = _masked_r2(preds, targets, ymask)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_losses": test_loss,
        "test_preds": preds,
        "test_targets": targets,
        "test_mask": ymask,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "model": model,
    }


if __name__ == "__main__":
    # Dictionary of molecules for molecular strings
    molecules = {
        "Methanol": "CO",
        "Ethanol": "CCO",
        "Benzene": "c1ccccc1",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    }

    sample_molecule = "Aspirin"
    aspirin_features, aspirin_adj, aspirin_edge_features, aspirin_edge_indices = (
        advanced_smiles_to_graph(molecules[sample_molecule])
    )
    logging.info(
        f"features, adjacency matrix, edge_features, edge_indices are as follows: {aspirin_features} {aspirin_adj} {aspirin_edge_features} {aspirin_edge_indices}"
    )

    # Training an example
    sample_data = mol_to_graph(molecules["Aspirin"])
    logging.info(sample_data, sample_data.x.shape[1])

    # implementing the GCNLayer
    model = GCNLayer(
        sample_data.x.shape[1], hidden_channels=64, out_channels=1, num_layers=4
    )

    dataset = MoleculeNet(root="data", name="ESOL")
    indices = torch.randperm(len(dataset))
    # Define the indicies for the train, validation and test indices
    train_idx = indices[: int(0.8 * len(dataset))]
    val_idx = indices[int(0.8 * len(dataset)) : int(0.9 * len(dataset))]
    test_idx = indices[int(0.9 * len(dataset)) :]

    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    logging.info(dataset)
    indices = torch.randperm(len(dataset))
    logging.info(indices)

    # train_and_evaluate()
    pooling_results = compare_pooling_methods(
        dataset, train_loader, val_loader, test_loader
    )
