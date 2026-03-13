"""Utilities for lightweight RDKit + scikit-learn examples.

This module used to execute demo code at import time. It now provides small,
reusable helper functions for generating Morgan fingerprints.
"""

from rdkit import Chem
from rdkit.Chem import AllChem


DEFAULT_SMILES = (
    "c1ccccc1",
    "c1ccccc1CC",
    "c1ccncc1",
    "c1ccncc1CC",
)


def mols_from_smiles(smiles_list=DEFAULT_SMILES):
    """Convert an iterable of SMILES strings into RDKit molecule objects."""
    return [Chem.MolFromSmiles(smiles) for smiles in smiles_list]


def morgan_fingerprints(mols, radius=2):
    """Create Morgan bit-vector fingerprints from a list of molecules."""
    return [AllChem.GetMorganFingerprintAsBitVect(mol, radius) for mol in mols if mol is not None]


def demo_fingerprints():
    """Generate demo fingerprints for a tiny built-in molecule set."""
    return morgan_fingerprints(mols_from_smiles())
