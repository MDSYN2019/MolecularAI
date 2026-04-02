# MolecularAI

![MolecularAI project logo](assets/molecularai-logo.svg)

This repository contains the **Chemistry2quant** codebase, with a mix of cheminformatics experiments, graph neural network work, and generated project documentation.

## Directory guide

### Top-level directories

- `.github/`
  GitHub automation, currently focused on CI workflows for repository checks and maintenance.

- `assets/`
  Static project assets (for example, the MolecularAI logo used in documentation).

- `docs/`
  Generated API/reference documentation artifacts (HTML and LaTeX outputs, plus Doxygen configuration files).

- `src/`
  Main Python source code organized by domain (`molecule`, `protein`, and `quant`).

- `test/`
  Test area reserved for validation scripts and experiment checks (currently minimal/placeholder).

### `src/` subdirectories

- `src/molecule/`
  PyTorch Geometric-focused molecular ML work: molecule featurization, graph construction from SMILES, GNN model/training implementations, hyperparameter tuning, and competition/inference utilities. Includes a dedicated learning-path README and sample ESOL data for experiments.

- `src/protein/`
  Early protein-graph exploration, currently a small Graphein configuration example used to bootstrap protein graph workflows.

- `src/quant/`
  Broader cheminformatics and quantum-adjacent toolkit area. This folder includes RDKit/scikit-learn/TensorFlow experiments, screening/QSAR utilities, molecule embedding work (Mol2Vec), database/pandas helpers, and scripts that bridge molecular structures into Psi4 input formats.

### Notable nested data/documentation directories

- `src/molecule/data/`
  Local experiment datasets/cache files used by molecule graph workflows (e.g., ESOL raw/processed artifacts).

- `src/quant/smifiles/`
  SMILES and related tabular input files used by quant/cheminformatics experiments.

- `docs/html/` and `docs/latex/`
  Built documentation outputs in browser-friendly and printable formats.

## Quick orientation

If you are getting back into the project, start with:

1. `src/molecule/README.md` for the cleanest end-to-end workflow explanation.
2. `src/quant/` for historical/experimental cheminformatics utilities.
3. `docs/html/index.html` for generated reference browsing.
