import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm as tqdm

# Performance debugging
# rdkit for molecular modelling
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import DataLoader
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

    - Initial GCN layer to process input node features
    - Optional multiple hidden GCN layers
    - Global mean pooling to get graph level representation
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int =2,
                 pooling: str = 'mean'):
        
        super(GCNLayer, self).__init__()


        # The input layer
        self.conv1 = GCNConv(in_channels, hidden_channels)

        # Additional hidden GCN layers: hidden_dim -> hidden_dim
        self.convs = nn.ModuleList()  # an initia list to build up the hidden neural network layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Simple 2 layer MLP head for graph-level predictions
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def _pool(self, x, batch):
        """
        apply different pooling methods depending on the definitions of the pooling defined
        in init 
        """
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling == "max":
            return global_max_pool(x, batch)
        else:
            return global_add_pool(x, batch)
        
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor) -> None:  # what to return here

        # The first convolutional layer 
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        # additional gcn layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        x = self._pool(x, batch)
        # multi level perceptron for final prediction
        x = F.relu(self.lin1(x))
        x = self.lin2(x) # [num_graphs, out_channels]
        return x



# Will be working on this next


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
