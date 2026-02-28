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
