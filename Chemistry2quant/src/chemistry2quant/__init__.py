"""Top-level package for chemistry2quant.

The package intentionally avoids importing optional heavy dependencies at import
-time (e.g., ``psi4`` and ``rdkit``). Import those libraries directly in the
submodules that require them.
"""

__all__ = [
    "chem2quant_analysis",
    "chem2quant_gen_psi4_input",
    "chem2quant_lipophilicity",
    "chem2quant_mol2vec",
    "chem2quant_mol2vec_ChEMBL",
    "chem2quant_neural_networks",
    "chem2quant_pandas",
    "chem2quant_scikit",
    "chem2quant_screening",
    "chem2quant_sdf2psi4",
    "chem2quant_structgen",
]
