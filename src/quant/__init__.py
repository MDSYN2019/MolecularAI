"""Top-level package for chemistry2quant.

The package intentionally avoids importing optional heavy dependencies at import
-time (e.g., ``psi4`` and ``rdkit``). Import those libraries directly in the
submodules that require them.
"""

__all__ = [
    "mol_analysis",
    "mol_gen_psi4_input",
    "mol_lipophilicity",
    "mol_mol2vec",
    "mol_mol2vec_ChEMBL",
    "mol_neural_networks",
    "mol_pandas",
    "mol_scikit",
    "mol_screening",
    "mol_sdf2psi4",
    "mol_structgen",
]
