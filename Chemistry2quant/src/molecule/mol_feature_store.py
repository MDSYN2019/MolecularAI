## rdkit
import pandas as pd
from rdkit.Chem import Descriptors

from mol_features import datamol_clean_cartridge

molecular_data_source = datamol_clean_cartridge()

# Use RDKit descriptors directly
molecular_data_source["mol_weight"] = molecular_data_source["mol"].apply(
    Descriptors.MolWt
)
molecular_data_source["logp"] = molecular_data_source["mol"].apply(Descriptors.MolLogP)
molecular_data_source["tpsa"] = molecular_data_source["mol"].apply(Descriptors.TPSA)

# Feast requires event_timestamp
molecular_data_source["event_timestamp"] = pd.Timestamp.now()

molecular_data_source[["id", "mol_weight", "logp", "tpsa", "event_timestamp"]].to_csv(
    "../../data/molecule_features.csv", index=False
)
