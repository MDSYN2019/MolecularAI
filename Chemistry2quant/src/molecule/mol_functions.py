# Performance debugging
# rdkit for molecular modelling
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

logging.basicConfig(level=logging.INFO)

PROPERTIES = ["Tg", "FFV", "Tc", "Density", "Rg"]

# labelling bond types
BOND_TYPE_TO_IDX = {
    rdchem.BondType.SINGLE: 0,
    rdchem.BondType.DOUBLE: 1,
    rdchem.BondType.TRIPLE: 2,
    rdchem.BondType.AROMATIC: 3,
}


STEREO_TO_IDX = {
    rdchem.BondStereo.STEREONONE: 0,
    rdchem.BondStereo.STEREOZ:    1,
    rdchem.BondStereo.STEREOE:    2,
    rdchem.BondStereo.STEREOANY:  3,
}


BOND_DIR_TO_IDX = {
    rdchem.BondDir.NONE:         0,
    rdchem.BondDir.BEGINWEDGE:   1,
    rdchem.BondDir.BEGINDASH:    2,
    rdchem.BondDir.ENDDOWNRIGHT: 3,
    rdchem.BondDir.ENDUPRIGHT:   4,
}

bond_type_to_idx = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}

HYB_CHOICES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
CHI_CHOICES = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
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


RD_FEATURES = [
    Descriptors.MolWt, Descriptors.HeavyAtomCount, Descriptors.NumValenceElectrons,
    Descriptors.NumAromaticRings, Descriptors.NumAliphaticRings, Descriptors.FractionCSP3,
    Descriptors.TPSA, Descriptors.MolMR, Descriptors.NumHAcceptors, Descriptors.NumHDonors,
    Descriptors.NumSaturatedRings, Descriptors.NumAliphaticCarbocycles, Descriptors.NumAromaticHeterocycles,
]

def rdkit_globals(smiles: str) -> np.ndarray:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.zeros(len(RD_FEATURES), dtype=np.float32)
    vals = [float(f(m)) for f in RD_FEATURES] # get the descriptors for each atom - these are listed on the RD_features
    return np.array(vals, dtype=np.float32)

def randomize_smiles(s):
    """
    Randomize the smiles
    """
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, doRandom=True, canonical=False) if m else s

def return_molecular_graph(adjacency, node_labels) -> nx.Graph:
    """
    """
    G = nx.Graph()
    for i, label in enumerate(node_labels):
        G.add_node(i, label=label)

    rows, cols = np.where(adjacency == 1)
    for i, j in zip(rows.tolist(), cols.tolist()):
        if i < j:
            G.add_edge(i, j)
    return G

def graph_descriptors(mol):
    """
    From the mol, return a torch graph 
    """
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
    if smiles_mol is None:
        raise ValueError(f"Invalid Smiles String: {smiles_mol}")

    mol = Chem.AddHs(smiles_mol)
    n_atoms = mol.GetNumAtoms()

    node_features = np.zeros((n_atoms, len(elements)), dtype=np.float32)
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        node_features[idx, elements.index(symbol) if symbol in elements else -1] = 1.0

    adjacency, edge_features, edge_indices = create_adjacency_matrix(n_atoms, mol)
    return node_features, adjacency, edge_features, edge_indices

def create_adjacency_matrix(n_atoms, mol):
    bond_type_to_idx = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    adjacency = np.zeros((n_atoms, n_atoms), dtype=np.float32)
    edge_features, edge_indices = [], []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[i, j] = adjacency[j, i] = 1.0

        feat = np.zeros(len(bond_type_to_idx), dtype=np.float32)
        bt = bond.GetBondType()
        if bt in bond_type_to_idx:
            feat[bond_type_to_idx[bt]] = 1.0

        # undirected → add both directions
        edge_features.append(feat); edge_indices.append((i, j))
        edge_features.append(feat); edge_indices.append((j, i))

    edge_features = np.array(edge_features, dtype=np.float32) if edge_features else np.empty((0, len(bond_type_to_idx)), dtype=np.float32)
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


def advanced_smiles_to_graph(smiles, bond_type_to_idx=bond_type_to_idx):
    """
    We have defined the bond_type_to_idx above as a general definition
    
    The simple representation above use only atom type and bond types, but real-world applications often need
    more sophisticated features.

    Convert a SMILES string to graph representation with advanced features.
    """
    if smiles is None:
        raise ValueError("Invalid SMILES string")

    
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    n_atoms = mol.GetNumAtoms()

    node_features = []
    # molecular_properties = get_molecular_properties(mol)
    for atom in mol.GetAtoms():
        atom_type = atom.GetSymbol()
        atomic_num = atom.GetAtomicNum()
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

    # Creating the adjacency matrix and edge features

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

        # Add edge in both directions (undirected graph)
        edge_features.append(features)
        edge_indices.append((begin_idx, end_idx))

        edge_features.append(features)  # Same feature for the reverse direction
        edge_indices.append((end_idx, begin_idx))

    # Convert edge features to numpy array
    if edge_features:
        edge_features = np.array(edge_features)
    else:
        edge_features = np.empty(
            (0, len(bond_type_to_idx) + 2)
        )  # +2 for conjugation and ring

    return node_features, adjacency, edge_features, edge_indices


def mol_to_graph(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    x = torch.stack([featurize_atom(a) for a in mol.GetAtoms()], dim=0)  # [N, Dx], float32
    ei, ea = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)  # [15], float32
        ei.append((i, j)); ea.append(bf)
        ei.append((j, i)); ea.append(bf)
    edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous() if ei else torch.empty((2, 0), dtype=torch.long)
    edge_attr  = torch.stack(ea, dim=0) if ea else torch.empty((0, 15), dtype=torch.float32)

    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    g.u = graph_descriptors(mol).view(1, -1)  # [1, U_raw]; remember to scale later
    return g



def atom_features(atom):
    """
    Extract a feature vector for an RDKit atom.

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
    # one-hot bond type (4)
    bt = [0,0,0,0]
    t = bond.GetBondType()
    if   t == Chem.rdchem.BondType.SINGLE:  bt[0]=1
    elif t == Chem.rdchem.BondType.DOUBLE:  bt[1]=1
    elif t == Chem.rdchem.BondType.TRIPLE:  bt[2]=1
    elif t == Chem.rdchem.BondType.AROMATIC:bt[3]=1
    # stereo (none, Z, E, any) -> 4
    st = [0,0,0,0]; s = int(bond.GetStereo())
    if s in (0,1,2,3): st[s]=1
    # direction (NONE, WEDGE, DASH, ENDDOWNRIGHT, ENDUPRIGHT) -> 5
    dir_map = {0:0,1:1,2:2,3:3,4:4}
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


# -------- load everything --------
#official = load_official(TRAIN_CSV)
#
#supp_frames = []
#
## dataset1.csv (Tc)
#supp_frames.append(load_tc_dataset("dataset1.csv"))
#
## the “MILES only” list (no labels) – skip for supervised training.
## If you want to use it for self-supervised pretraining, keep it separate.
#
## dataset3.csv (Tg)
#supp_frames.append(load_tg_dataset("dataset3.csv"))
#
## dataset4.csv (FFV) – adjust header_has_names if yours has no header line
#supp_frames.append(load_ffv_dataset("dataset4.csv", header_has_names=False))
#
## concat
#big = pd.concat([official] + supp_frames, ignore_index=True, sort=False)
#
## canonicalize SMILES and drop rows that fail to parse
#big["SMILES"] = big["SMILES"].map(canon_polymer_smiles)
#big = big[big["SMILES"].notna()].copy()
#
## resolve duplicates/conflicts at SMILES level
#merged = (
#    big.groupby("SMILES", as_index=True)
#       .apply(resolve_conflicts)
#       .reset_index(drop=True)
#)
#
## if you still want to keep IDs from official rows, you can merge back:
## merged = merged.merge(official[["SMILES","id"]], on="SMILES", how="left")

# final sanity checks
# clip FFV to [0,1], drop impossible densities, etc.
#if "FFV" in merged.columns:
#    merged.loc[merged["FFV"].notna(), "FFV"] = merged["FFV"].clip(0.0, 1.0)
#if "Density" in merged.columns:
#    merged.loc[merged["Density"].notna(), "Density"] = merged["Density"].clip(lower=0.3, upper=3.0)
#
## optional: run fix_tg_units once more on the combined table
#merged = fix_tg_units(merged, tg_col="Tg")
#
## you can now use `merged` as your training table
## and build PyG graphs:
#from torch_geometric.data import Data
#merged["graph"] = merged["SMILES"].apply(mol_to_graph)

# then proceed with your existing split + attach_multitask_targets pipeline
