import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm as tqdm

# Performance debugging
# rdkit for molecular modelling
from rdkit import Chem
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    GINEConv,
    global_mean_pool,
    GlobalAttention,
    global_add_pool,
    global_max_pool,
)

logging.basicConfig(level=logging.INFO)

from mol_functions import (
    advanced_smiles_to_graph,
    mol_to_graph,
)


class GCNLayer(nn.Module):
    """
    Basic graph convolution network for graph-level prediction

    - Input GCN Layer
    - Optional hidden GCN layers
    - Global pooling -> graph embedding
    - MLP for final prediction
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()

        # The first GCN layer
        self.conv1 = GCNConv(in_channels, hidden_channels)

        # Additional GCN Layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers - 1): # add num_layers - 1 hidden layers
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # MLP head
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch) -> None:
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        # additional gcn layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        # multi level perceptron for final prediction
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x


class GINLayer(nn.Module):
    """
    Graph Isomorphism Network for graph-level regression.

    - MLP-based GINConv layers (num_layers controls the number of convs)
    - BatchNorm and ReLU after each convolution
    - Global add pooling to obtain a graph embedding
    - Two linear layers with optional dropout for final prediction
    """

    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers=10, dropout=0.5
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # first GIN layer takes in_channels -> hidden_channels
        mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.convs.append(GINConv(mlp))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # subsequent GIN layers keep hidden size
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # readout / prediction MLP
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # message passing
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        # graph-level pooling (sum, mean or max)
        x = global_mean_pool(x, batch)
        # max pool
        # x = global_max_pool(x, batch)
        # sum pool
        # x = global_add_pool(x, batch)
        # optional dropout to improve generalisation
        x = F.dropout(x, p=self.dropout, training=self.training)

        # final MLP
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x


class GINELayer(nn.Module):
    """
    Edge-aware GIN for graph-level regression.

    Need to get a breakdown on this
    """

    def __init__(
        self,
        in_channels,
        edge_dim,
        out_channels,
        hidden_channels=128,
        num_layers=6,
        dropout=0.2,
        pooling="sum",
    ):
        super().__init__()
        assert pooling in {"mean", "sum", "max"}
        self.pooling = pooling
        self.convs, self.bns = nn.ModuleList(), nn.ModuleList()

        def node_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
            )

        # First layer: x_dim -> hidden
        self.convs.append(
            GINEConv(node_mlp(in_channels, hidden_channels), edge_dim=edge_dim)
        )
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers: hidden -> hidden
        for _ in range(num_layers - 1):
            self.convs.append(
                GINEConv(node_mlp(hidden_channels, hidden_channels), edge_dim=edge_dim)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Readout head - I think this is the nonlinear learning feature per graph
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def _pool(self, x, batch):
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling == "sum":
            from torch_geometric.nn import global_add_pool

            return global_add_pool(x, batch)
        else:
            from torch_geometric.nn import global_max_pool

            return global_max_pool(x, batch)

    def forward(self, x, edge_index, edge_attr, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)

        x = self._pool(x, batch)
        return self.mlp_out(x)

class GINELayerUpgraded(nn.Module):
    """
    GINE backbone with practical upgrades:
      - Edge encoder (shared)
      - train_eps in GINEConv
      - Residual + Dropout between layers
      - Optional GRU node updates (AttentiveFP-style stability)
      - Readouts: attention / mean / sum / max / meanmax (mean+max concat)
      - LayerNorm before the head
    """

    def __init__(
        self,
        in_channels,
        edge_dim,
        out_channels,
        hidden_channels=128,
        num_layers=4,
        dropout=0.4,
        input_feature_dropout=0.1,
        edge_feature_dropout=0.05,
        pooling="attn",  # "attn" | "mean" | "sum" | "max" | "meanmax"
        use_gru=False,
        use_residual=True,
        norm="batch",  # "batch" | "layer" | None
        global_feat_dim=0,
    ):
        super().__init__()
        assert pooling in {"attn", "mean", "sum", "max", "meanmax"}
        assert norm in {"batch", "layer", None}
        self.pooling = pooling
        self.use_gru = use_gru
        self.use_residual = use_residual
        self.dropout_p = dropout
        self.input_feature_dropout = input_feature_dropout
        self.edge_feature_dropout = edge_feature_dropout
        self.global_feat_dim = global_feat_dim
        self.hidden_channels = hidden_channels

        # Optional global features projection (kept as additive to keep dims simple)
        self.u_proj = nn.Linear(self.global_feat_dim, hidden_channels) if self.global_feat_dim > 0 else None

        # ---- (A) Shared edge encoder ----
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # ---- (B) GINEConv stack with train_eps=True ----
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        def node_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.SiLU(),
                nn.Linear(out_dim, out_dim),
            )

        # First layer: in -> hidden
        self.convs.append(
            GINEConv(
                node_mlp(in_channels, hidden_channels),
                edge_dim=hidden_channels,
                train_eps=True,
            )
        )
        self.norms.append(self._make_norm(norm, hidden_channels))

        # Hidden layers: hidden -> hidden
        for _ in range(num_layers - 1):
            self.convs.append(
                GINEConv(
                    node_mlp(hidden_channels, hidden_channels),
                    edge_dim=hidden_channels,
                    train_eps=True,
                )
            )
            self.norms.append(self._make_norm(norm, hidden_channels))

        # ---- (C) Optional GRU cell ----
        if use_gru:
            self.gru = nn.GRU(hidden_channels, hidden_channels)

        # ---- (D) Readout ----
        if pooling == "attn":
            gate_nn = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.SiLU(),
                nn.Linear(hidden_channels // 2, 1),
            )
            self.readout = GlobalAttention(gate_nn=gate_nn)
            readout_dim = hidden_channels
        elif pooling == "meanmax":
            # concat(mean, max) -> 2H
            self.readout = None
            readout_dim = 2 * hidden_channels
        else:
            self.readout = None
            readout_dim = hidden_channels

        # (Optional) additive global features keep the readout dim the same
        # since we project u to H and add to g (no concat).

        # ---- (E) Pre-head norm + head ----
        self.pre_head_norm = nn.LayerNorm(readout_dim)
        self.mlp_out = nn.Sequential(
            nn.Linear(readout_dim, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    @staticmethod
    def _make_norm(norm, dim):
        if norm == "batch":
            return nn.BatchNorm1d(dim)
        elif norm == "layer":
            return nn.LayerNorm(dim)
        else:
            return nn.Identity()

    def _pool_basic(self, x, batch):
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling == "sum":
            return global_add_pool(x, batch)
        elif self.pooling == "max":
            return global_max_pool(x, batch)
        else:
            raise RuntimeError("Invalid basic pooling type")

    def _pool_meanmax(self, x, batch):
        h_mean = global_mean_pool(x, batch)
        h_max  = global_max_pool(x, batch)
        return torch.cat([h_mean, h_max], dim=-1)  # [B, 2H]

    def forward(self, x, edge_index, edge_attr, batch, u=None):
        # Input feature dropout
        if self.input_feature_dropout and self.training:
            x = F.dropout(x, p=self.input_feature_dropout, training=True)

        # Edge feature dropout (pre-encode)
        if self.edge_feature_dropout and self.training:
            drop_mask = torch.rand(edge_attr.size(0), device=edge_attr.device) < self.edge_feature_dropout
            edge_attr = edge_attr.clone()
            edge_attr[drop_mask] = 0.0

        enc_edge = self.edge_encoder(edge_attr)  # [E, H]
        h_prev = None

        # Backbone
        for conv, norm in zip(self.convs, self.norms):
            h_in = x
            x = conv(x, edge_index, enc_edge)     # [N, H]
            x = norm(x)
            x = F.silu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
            if self.use_gru:
                if h_prev is None:
                    h_prev = x.new_zeros(1, x.size(0), self.hidden_channels)
                x, h_prev = self.gru(x.unsqueeze(0), h_prev)
                x = x.squeeze(0)
            if self.use_residual and h_in.shape == x.shape:
                x = x + h_in

        # Readout
        if self.pooling == "attn":
            g = self.readout(x, batch)                    # [B, H]
        elif self.pooling == "meanmax":
            g = self._pool_meanmax(x, batch)              # [B, 2H]
        else:
            g = self._pool_basic(x, batch)                # [B, H]

        # Optional global features (additive, projected to H)
        if self.u_proj is not None:
            if u is None:
                u = torch.zeros(g.size(0), self.global_feat_dim, device=g.device)
            elif u.dim() == 1:
                u = u.unsqueeze(0).expand(g.size(0), -1)
            # If meanmax, add u only to the first half (mean) by splitting, or add to both halves.
            if self.pooling == "meanmax":
                H = self.hidden_channels
                u_proj = self.u_proj(u.to(g.device))      # [B, H]
                g = torch.cat([g[:, :H] + u_proj, g[:, H:]], dim=-1)
            else:
                g = g + self.u_proj(u.to(g.device))       # [B, H]

        # LayerNorm before head
        g = self.pre_head_norm(g)

        return self.mlp_out(g)
