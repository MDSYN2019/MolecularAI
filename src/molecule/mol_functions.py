# Performance debugging
# rdkit for molecular modelling
from mol_functions_chem import featurize_molecule_atoms
import optuna

import logging
import pandas as pd
import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdchem

from torch_geometric.data import Data
from rdkit.Chem import Descriptors, rdMolDescriptors

from sklearn.preprocessing import StandardScaler


PROPERTIES = ["Tg", "FFV", "Tc", "Density", "Rg"]
logging.basicConfig(level=logging.INFO)

# Define type of bond
BOND_TYPE_TO_IDX = {
    rdchem.BondType.SINGLE: 0,
    rdchem.BondType.DOUBLE: 1,
    rdchem.BondType.TRIPLE: 2,
    rdchem.BondType.AROMATIC: 3,
}

# Define type of stereochemistry
STEREO_TO_IDX = {
    rdchem.BondStereo.STEREONONE: 0,
    rdchem.BondStereo.STEREOZ:    1,
    rdchem.BondStereo.STEREOE:    2,
    rdchem.BondStereo.STEREOANY:  3,
}

# define type of bond dir
BOND_DIR_TO_IDX = {
    rdchem.BondDir.NONE:         0,
    rdchem.BondDir.BEGINWEDGE:   1,
    rdchem.BondDir.BEGINDASH:    2,
    rdchem.BondDir.ENDDOWNRIGHT: 3,
    rdchem.BondDir.ENDUPRIGHT:   4,
}

# define type of hybridization
HYB_CHOICES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

# define types of chiral atoms
CHI_CHOICES = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]

# define the sterochemistry
STEREO_CHOICES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOANY,
]

BOND_DIR_CHOICES = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.BEGINWEDGE,
    Chem.rdchem.BondDir.BEGINDASH,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
    Chem.rdchem.BondDir.ENDUPRIGHT,
]

# Static tables (trim as needed)
PAULING = {
    1: 2.20,
    6: 2.55,
    7: 3.04,
    8: 3.44,
    9: 3.98,
    15: 2.19,
    16: 2.58,
    17: 3.16,
    35: 2.96,
    53: 2.66,
}
COV_RAD = {
    1: 0.31,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    35: 1.20,
    53: 1.39,
}


# Basic descriptors we wish to compute for each smiles string  - global features to be added in 
RD_FEATURES = [
    Descriptors.MolWt, # molecular weight 
    Descriptors.HeavyAtomCount, # number of non-hydogen atoms 
    Descriptors.NumValenceElectrons, # total number of valence electrons in the molecule  
    Descriptors.NumAromaticRings, # number of aromatic rings in the molecule 
    Descriptors.NumAliphaticRings, # number of aliphatic rings in the molecule 
    Descriptors.FractionCSP3, # fraction of sp3-hybridized carbons 
    Descriptors.TPSA, # topological polar surface area
    Descriptors.MolMR, # molar refractivity 
    Descriptors.NumHAcceptors, # number of hydogen bond acceptors 
    Descriptors.NumHDonors, # number of hydrogen bond donors 
    Descriptors.NumSaturatedRings, # number of saturated rings in the molcule 
    Descriptors.NumAliphaticCarbocycles, # number of aliphatic carbocycles in the molecule 
    Descriptors.NumAromaticHeterocycles, # number of aromatic heterocycles in the molecule 
]

def rdkit_globals(smiles: str) -> np.ndarray:
    "Compute a compact vector of global RDKit descriptors for one molecule"
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.zeros(len(RD_FEATURES), dtype=np.float32)
    vals = [float(f(m)) for f in RD_FEATURES] # implement each feature for the smiles string
    return np.array(vals, dtype=np.float32)

def randomize_smiles(s):
    """Generate a randomized (non-canonical) SMILES augmentation when possible."""
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, doRandom=True, canonical=False) if m else s

def return_molecular_graph(adjacency, node_labels) -> nx.Graph:
    """Convert adjacency + labels into a simple NetworkX undirected graph."""
    G = nx.Graph()
    for i, label in enumerate(node_labels):
        G.add_node(i, label=label)

    rows, cols = np.where(adjacency == 1)
    for i, j in zip(rows.tolist(), cols.tolist()):
        if i < j:
            G.add_edge(i, j)
    return G

def graph_descriptors(mol) -> torch.Tensor:
    """Graph-level descriptors used as global conditioning features."""
    return torch.tensor([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        mol.GetNumHeavyAtoms(),
        sum(1 for a in mol.GetAtoms() if a.GetSymbol()=='F'),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum()==0),  # count '*'
    ], dtype=torch.float32)


def featurize_atom(atom):
    """
    Per atom, get the atomic number, and featurize the hybridization etc
    """
    Z = atom.GetAtomicNum()
    # categorical one-hots
    hyb = atom.GetHybridization()
    hyb_vec = [
        int(hyb == rdchem.HybridizationType.SP),
        int(hyb == rdchem.HybridizationType.SP2),
        int(hyb == rdchem.HybridizationType.SP3),
        int(hyb == rdchem.HybridizationType.SP3D),
        int(hyb == rdchem.HybridizationType.SP3D2),
    ]
    chi = atom.GetChiralTag()
    chi_vec = [
        int(chi == rdchem.ChiralType.CHI_UNSPECIFIED),
        int(chi == rdchem.ChiralType.CHI_TETRAHEDRAL_CW),
        int(chi == rdchem.ChiralType.CHI_TETRAHEDRAL_CCW),
        int(chi == rdchem.ChiralType.CHI_OTHER),
    ]
    # scalar physchem
    en = float(PAULING.get(Z, np.mean(list(PAULING.values()))))
    cov = float(COV_RAD.get(Z, np.mean(list(COV_RAD.values()))))

    # polymer anchor (‘*’) detection: RDKit uses atomic number 0 for wildcard
    is_anchor = float(Z == 0.0)

    feats = [
        float(Z),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        float(atom.GetDegree()),
        float(atom.GetNumRadicalElectrons()),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        en, cov, is_anchor,
    ] + hyb_vec + chi_vec

    return torch.tensor(feats, dtype=torch.float32)


def smiles_to_graph(smiles_mol, elements=["C", "O", "N", "H", "Other"]):
    """
    Build a lightweight one-hot atom graph representation from an RDKit molecule.

    Returns node one-hots, adjacency matrix, edge one-hots and directed edge index tuples.
    """
    if smiles_mol is None:
        raise ValueError(f"Invalid Smiles String: {smiles_mol}")

    mol = Chem.AddHs(smiles_mol)
    n_atoms = mol.GetNumAtoms()

    # Create container for the node features
    node_features = np.zeros((n_atoms, len(elements)), dtype=np.float32)

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        node_features[idx, elements.index(symbol) if symbol in elements else -1] = 1.0

    adjacency, edge_features, edge_indices = create_adjacency_matrix(n_atoms, mol)
    return node_features, adjacency, edge_features, edge_indices

def _append_undirected_edge(edge_features, edge_indices, feat, i, j):
    """Store an undirected bond as two directed edges for PyG compatibility."""
    edge_features.append(feat)
    edge_indices.append((i, j))
    edge_features.append(feat)
    edge_indices.append((j, i))


def create_adjacency_matrix(n_atoms, mol):
    """Create adjacency and one-hot bond features from an RDKit molecule."""
    adjacency = np.zeros((n_atoms, n_atoms), dtype=np.float32)
    edge_features, edge_indices = [], []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[i, j] = adjacency[j, i] = 1.0

        feat = np.zeros(len(BOND_TYPE_TO_IDX), dtype=np.float32)
        bt = bond.GetBondType()
        if bt in BOND_TYPE_TO_IDX:
            feat[BOND_TYPE_TO_IDX[bt]] = 1.0

        _append_undirected_edge(edge_features, edge_indices, feat, i, j)

    edge_features = np.array(edge_features, dtype=np.float32) if edge_features else np.empty((0, len(BOND_TYPE_TO_IDX)), dtype=np.float32)
    return adjacency, edge_features, edge_indices


def get_molecular_properties(mol):
    """
    return dictionary for properties for the following properties:

    - Atom type
    - Atomic number
    - Formal charge
    - Hybridization
    - Whether it is aromatic
    - Whether it is in a ring

    """
    property_dictionary = {}
    properties = [
        "atom_type",
        "atomic_num",
        "formal_charge",
        "hybridization",
        "is_aromatic",
        "is_in_ring",
    ]
    property_values = []

    # Get the properties
    for atom in mol.GetAtoms():
        atom_type = atom.GetSymbol()
        atomic_num = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge()
        hybridization = atom.GetHybridization()
        is_aromatic = int(atom.GetIsAromatic())
        is_in_ring = int(atom.IsInRing())
        property_values = [
            atom_type,
            atomic_num,
            formal_charge,
            hybridization,
            is_aromatic,
            is_in_ring,
        ]

    for property, value in zip(properties, property_values):
        property_dictionary[property] = value

    return property_dictionary


def smiles_to_pytorch_graph(smiles):
    """
    Pytorch geometric for molecular graphs

    Now that we understand the fundamentals of graph representation for molecules, let's implement this using Pytorch Geometric (PyG),
    a library specifically designed for graph neural networks
    """
    # Get the graph representation
    node_features, adjacency, edge_features, edge_indices = advanced_smiles_to_graph(
        smiles
    )

def mol_to_graph(smiles: str) -> Data:
    """Convert SMILES into a PyG Data object with atom, bond and global features."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    x = torch.stack([featurize_atom(a) for a in mol.GetAtoms()], dim=0)  # [N, Dx], float32
    ei, ea = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx() # get edge indices
        bf = bond_features(bond)  # [15], float32  - create the edge features
        ei.append((i, j)); ea.append(bf) # append bond feature one way
        ei.append((j, i)); ea.append(bf) # append bond feature the other way

    edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous() if ei else torch.empty((2, 0), dtype=torch.long)
    edge_attr  = torch.stack(ea, dim=0) if ea else torch.empty((0, 15), dtype=torch.float32)

    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    g.u = graph_descriptors(mol).view(1, -1)  # [1, U_raw]; remember to scale later
    return g

def atom_features(atom):
    """
    Extract a feature vector for an RDKit atom - per atom node features to be added in

    Features included:
        - Atomic number
        - Chirality tag (encoded as integer)
        - Degree (number of directly-bonded atoms)
        - Formal charge
        - Total number of hydrogens
        - Number of radical electrons
        - Hybridization (encoded as integer)
        - Aromaticity (0 or 1)
        - Ring membership (0 or 1)

    Args:
        atom (rdkit.Chem.rdchem.Atom): An RDKit Atom object.

    Returns:
        torch.Tensor: Feature tensor of shape (9,) with dtype long.
    """
    return torch.tensor(
        [
            atom.GetAtomicNum(),  # Atomic number
            int(atom.GetChiralTag()),  # Chirality
            atom.GetDegree(),  # Degree
            atom.GetFormalCharge(),  # Formal charge
            atom.GetTotalNumHs(),  # Number of hydrogens
            atom.GetNumRadicalElectrons(),  # Radical electrons
            int(atom.GetHybridization()),  # Hybridization
            int(atom.GetIsAromatic()),  # Aromaticity
            int(atom.IsInRing()),  # Ring membership
        ],
        dtype=torch.long,
    )


def bond_features(bond):
    """
    Depending on the bond type we do detect,
    one-hot vectorize the vector
    """
    # one-hot bond type (4)
    bt = [0,0,0,0]
    t = bond.GetBondType()
    if   t == Chem.rdchem.BondType.SINGLE:  bt[0]=1
    elif t == Chem.rdchem.BondType.DOUBLE:  bt[1]=1
    elif t == Chem.rdchem.BondType.TRIPLE:  bt[2]=1
    elif t == Chem.rdchem.BondType.AROMATIC:bt[3]=1
    # stereo (none, Z, E, any) -> 4
    st = [0,0,0,0]; s = int(bond.GetStereo()) # store in sterotype one-hot vector
    if s in (0,1,2,3): st[s]=1
    # direction (NONE, WEDGE, DASH, ENDDOWNRIGHT, ENDUPRIGHT) -> 5
    dir_map = {0:0,1:1,2:2,3:3,4:4} # direction
    d = [0,0,0,0,0]; di = dir_map.get(int(bond.GetBondDir()),0); d[di]=1
    # flags
    flags = [int(bond.GetIsConjugated()), int(bond.IsInRing())]
    return torch.tensor(bt + st + d + flags, dtype=torch.float32)  # dim = 4+4+5+2=15

def fix_tg_units(
    df: pd.DataFrame,
    tg_col: str = "Tg",
    flag_col: str = "Tg_was_kelvin",
    assume_kelvin_if_gt: float = 200.0,
) -> pd.DataFrame:
    """
    Convert Tg from K→°C where needed.
    Rules:
      1) If flag_col == 1 -> convert (K to °C).
      2) Also convert if Tg looks like Kelvin by magnitude (e.g. > 200 K and < 1000 K).
      3) Leave everything else (including NaNs) untouched.
    Adds/updates flag_col to 1 where a conversion happened.
    """
    out = df.copy()

    if flag_col not in out.columns:
        out[flag_col] = 0.0

    tg = out[tg_col]
    # explicit flag
    mask_flag = out[flag_col].fillna(0).astype(int).eq(1)
    # heuristic (typical Tg(K) are >200; Celsius Tg rarely > 200)
    mask_guess = tg.notna() & (tg > assume_kelvin_if_gt) & (tg < 1000)

    mask_convert = mask_flag | mask_guess

    # convert K -> °C
    out.loc[mask_convert, tg_col] = tg[mask_convert] - 273.15
    # mark rows we converted
    out.loc[mask_convert, flag_col] = 1.0

    # optional: assert plausible Tg(°C) range
    plausible = out[tg_col].isna() | ((out[tg_col] > -200) & (out[tg_col] < 300))
    if not plausible.all():
        bad = out.loc[~plausible, [tg_col, flag_col, "id"]]
        raise ValueError(f"Unplausible Tg after conversion; inspect:\n{bad.head()}")

    return out



def advanced_smiles_to_graph(smiles: str, bond_type_to_idx=BOND_TYPE_TO_IDX):
    """
    The simple representation above use only atom type and bond types, but real-world applications often need
    more sophisticated features.

    Convert a SMILES string to graph representation with advanced features.

    This is a developed version of all the feature building functions per node and per bond that we have made above

    """
    if smiles is None:
        raise ValueError("Invalid SMILES string")

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    n_atoms = mol.GetNumAtoms()

    # --------------- CREATING NODE LEVEL FEATURES ---------------------
    node_features = []
    # molecular_properties = get_molecular_properties(mol)
    for atom in mol.GetAtoms():
        atom_type = atom.GetSymbol() # get the atom type
        atomic_num = atom.GetAtomicNum() # get the atomic number
        formal_charge = atom.GetFormalCharge()
        hybridization = atom.GetHybridization()
        is_aromatic = int(atom.GetIsAromatic())
        is_in_ring = int(atom.IsInRing())
        logging.info(
            f"{atom_type} {atomic_num} {formal_charge} {hybridization} {is_aromatic} {is_in_ring}"
        )
        # Create one-hot encoding for atom type (C, O, N, H, F, S, ..)
        atom_types = ["C", "O", "N", "H", "F", "P", "S", "Cl", "Br", "I"]
        atom_type_onehot = [1 if atom_type == t else 0 for t in atom_types]

        # If we don't have this atom type, then we need to account for this new atomic entry
        if atom_type not in atom_types:
            atom_type_onehot.append(1)
        else:
            atom_type_onehot.append(0)

        # One-hot for hybridization

        hybridization_types = [
            Chem.rdchem.HybridizationType.SP,  # Which type of hybridization is this?
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
        ]
        hybridization_onehot = [
            1 if hybridization == h else 0 for h in hybridization_types
        ]
        logging.info(
            f"For the molecule we get the following hydbridizations {hybridization_onehot}"
        )

        if hybridization not in hybridization_types:
            hybridization_onehot.append(1)  # "Other" hybridization
        else:
            hybridization_onehot.append(0)
            logging.info(
                f"one hot {atom_type_onehot}, hybridization {hybridization_onehot}"
            )

        # Combine all features

        features = (
            atom_type_onehot
            + [
                formal_charge,
                is_aromatic,
                is_in_ring,
                atom.GetDegree(),
                atom.GetTotalNumHs(),
                atom.GetNumRadicalElectrons(),
            ]
            + hybridization_onehot
        )

        node_features.append(features)
        logging.info(f"The final feature we get is {features}")

    # Convert to numpy array
    node_features = np.array(node_features)
    logging.info(f"The final node features we have combined is: {node_features}")

    # ------------- CREATING THE ADJACENCY MATRIX AND EDGE FEATURES ----------------

    adjacency = np.zeros((n_atoms, n_atoms))
    edge_features = []
    edge_indices = []

    for bond in mol.GetBonds():
        # Get the atoms in the bond
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        # Update adjacency matrix (symmetric)
        adjacency[begin_idx, end_idx] = 1
        adjacency[end_idx, begin_idx] = 1

        # Advanced bond features
        bond_type = bond.GetBondType()
        bond_type_onehot = np.zeros(len(bond_type_to_idx))
        if bond_type in bond_type_to_idx:
            bond_type_onehot[bond_type_to_idx[bond_type]] = 1

        is_conjugated = int(bond.GetIsConjugated())
        is_in_ring = int(bond.IsInRing())

        # Combine all bond features
        features = np.concatenate([bond_type_onehot, [is_conjugated, is_in_ring]])

        _append_undirected_edge(
            edge_features,
            edge_indices,
            features,
            begin_idx,
            end_idx,
        )

    # Convert edge features to numpy array
    if edge_features:
        edge_features = np.array(edge_features)
    else:
        edge_features = np.empty(
            (0, len(bond_type_to_idx) + 2)
        )  # +2 for conjugation and ring

    return node_features, adjacency, edge_features, edge_indices

# Other boilerplate functions to read in the supplementary information
# -------- helpers --------
def canon_polymer_smiles(s: str) -> str | None:
    """Canonicalize while preserving '*' wildcard atoms."""
    if not isinstance(s, str) or not s.strip():
        return None
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    return Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)

def load_official(train_csv: str) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    # normalize column names
    df = df.rename(columns={"d":"id"})  # your header shows 'd'
    # keep only known columns
    keep = ["id","SMILES"] + PROPERTIES
    df = df[[c for c in keep if c in df.columns]].copy()
    df["source"] = "official"
    df["official"] = True
    return df

def load_tc_dataset(path: str) -> pd.DataFrame:
    """dataset1.csv: columns like MILES,TC_mean"""
    df = pd.read_csv(path)
    df = df.rename(columns={"MILES":"SMILES","TC_mean":"Tc"})
    df = df[["SMILES","Tc"]].copy()
    df["source"] = "supp_tc"
    df["official"] = False
    return df

def load_tg_dataset(path: str) -> pd.DataFrame:
    """dataset3.csv: SMILES,Tg (may be K/°C mixed)"""
    df = pd.read_csv(path)
    df = df.rename(columns={"MILES":"SMILES"})
    df = df[["SMILES","Tg"]].copy()
    # apply your Kelvin→°C fixer
    df = fix_tg_units(df, tg_col="Tg")
    df["source"] = "supp_tg"
    df["official"] = False
    return df

def load_ffv_dataset(path: str, header_has_names: bool = True) -> pd.DataFrame:
    """dataset4.csv: likely SMILES,FFV (values ~0.35-0.39)"""
    if header_has_names:
        df = pd.read_csv(path)
        # try to detect column names
        if "FFV" not in df.columns:
            # assume 2 columns present
            cols = list(df.columns)
            if len(cols) >= 2:
                df = df.rename(columns={cols[0]:"SMILES", cols[1]:"FFV"})
    else:
        df = pd.read_csv(path, header=None, names=["SMILES","FFV"])
    df = df[["SMILES","FFV"]].copy()
    df["source"] = "supp_ffv"
    df["official"] = False
    return df

def _to_num(s: pd.Series) -> pd.Series:
    # strip brackets/whitespace just in case, then coerce
    s = s.astype(str).str.strip().str.strip("[]")
    return pd.to_numeric(s, errors="coerce")

def resolve_conflicts(group: pd.DataFrame) -> pd.Series:
    out = {}
    # ... whatever keys you iterate over, e.g. PROPERTIES ...
    for p in PROPERTIES:
        vals_all = _to_num(group[p]).dropna()
        if "official" in group.columns:
            vals_off = _to_num(group.loc[group["official"] == True, p]).dropna()
        else:
            vals_off = pd.Series(dtype=float)

        if len(vals_off):
            out[p] = vals_off.median()
        elif len(vals_all):
            out[p] = vals_all.median()
        else:
            out[p] = pd.NA
    # return a Series so GroupBy.apply can stack resu
    return pd.Series(out)

def normalize_adjacency(A):
    """
    Returns the symmetrically normalized adjacency matrix

    This normalizes the adjacency matrix by accounting for node degrees, helping
    prevent bias from high-degree nodes. It uses the common normalization:

    A_hat = D^(-1/2)(A + I) D^(-1/2)

    Where A is the adjacency matrix, I is the identity, and D is thr degree matrix
    """
    I = torch.eye(A.size(0))  # Get the identity matrix
    A_hat = A + I  # add the identity matrix to the adjacency matrix
    D_hat = torch.diag(torch.pow(A_hat.sum(1), -0.5))  # return the normalized adjacency
    return D_hat @ A_hat @ D_hat

def align_targets(
    pred: torch.Tensor, y: torch.Tensor, y_mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ensure y and y_mask match pred shape [B, T].
    Accepts y / mask as [B], [B*T], or already [B, T].
    """
    B, T = pred.shape

    # --- fix y ---
    if y is None:
        raise RuntimeError("Target y is None")
    if y.dim() == 1:  # [B] or [B*T]
        if y.numel() == B * T:
            y = y.view(B, T)
        elif y.numel() == B:
            y = y.view(B, 1).expand(B, T)
        else:
            raise RuntimeError(
                f"Cannot reshape y of shape {tuple(y.shape)} to [B,T]=[{B},{T}]"
            )
    elif y.dim() == 2 and (y.shape[0] == B and y.shape[1] == T):
        pass  # good
    else:
        # last resort: same numel?
        if y.numel() == B * T:
            y = y.view(B, T)
        else:
            raise RuntimeError(
                f"Incompatible y shape {tuple(y.shape)} for pred {tuple(pred.shape)}"
            )

    # --- fix mask ---
    if y_mask is None:
        y_mask = torch.ones_like(y, dtype=y.dtype, device=y.device)
    else:
        if y_mask.dim() == 1:  # [B] or [B*T]
            if y_mask.numel() == B * T:
                y_mask = y_mask.view(B, T)
            elif y_mask.numel() == B:
                y_mask = y_mask.view(B, 1).expand(B, T)
            else:
                raise RuntimeError(
                    f"Cannot reshape mask {tuple(y_mask.shape)} to [B,T]=[{B},{T}]"
                )
        elif y_mask.dim() == 2 and (y_mask.shape[0] == B and y_mask.shape[1] == T):
            pass
        else:
            if y_mask.numel() == B * T:
                y_mask = y_mask.view(B, T)
            else:
                raise RuntimeError(
                    f"Incompatible mask shape {tuple(y_mask.shape)} for pred {tuple(pred.shape)}"
                )

        y_mask = y_mask.to(y.device, dtype=y.dtype)

    return y, y_mask
