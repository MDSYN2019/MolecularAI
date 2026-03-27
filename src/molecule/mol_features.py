"""Utilities for loading eMolecules from a PostgreSQL RDKit cartridge and deriving features."""

import asyncio
import os
from dataclasses import dataclass, field
from math import asin, cos, radians, sin, sqrt
from typing import Iterable

import asyncpg
import datamol as dm
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


async def connect_to_db(default_query: str = "SELECT * FROM raw_data"):
    """Connect to the rdkit cartridge database and run a query."""
    pool = await asyncpg.create_pool(
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        database="emolecules",
        host="localhost",
        port=5432,
        min_size=1,
        max_size=10,
    )
    try:
        async with pool.acquire() as conn:
            result = await conn.fetch(default_query)
            return result
    finally:
        await pool.close()


async def execute_molecular_filter(
    smiles_query: str = "SELECT id, structure FROM molecules WHERE structure@>'c1cccnc1' LIMIT 100;",
) -> Iterable[asyncpg.Record]:
    """Return rows from a cartridge query (defaults to pyridine-like substructure)."""
    results = await connect_to_db(smiles_query)
    return results or []


async def list_tables():
    """Debug helper to inspect available tables in PostgreSQL."""
    result = await connect_to_db(
        "SELECT table_schema, table_name FROM information_schema.tables"
    )
    for row in result:
        print(dict(row))


def preprocess_postgres_cartridge(
    query: str = "SELECT id, structure FROM molecules WHERE structure@>'c1cccnc1' LIMIT 100;",
) -> pd.DataFrame:
    """Load query results from the PostgreSQL RDKit cartridge into a DataFrame."""
    cartridge_result = asyncio.run(execute_molecular_filter(query))
    output_data = [{"id": row["id"], "structure": row["structure"]} for row in cartridge_result]
    return pd.DataFrame(output_data)


def datamol_clean_cartridge(
    query: str = "SELECT id, structure FROM molecules WHERE structure@>'c1cccnc1' LIMIT 100;",
) -> pd.DataFrame:
    """Load cartridge records and create a sanitized RDKit molecule column."""
    dataframe = preprocess_postgres_cartridge(query=query)
    if dataframe.empty:
        return dataframe

    dataframe["mol"] = dataframe["structure"].apply(dm.to_mol)
    dataframe = dataframe[dataframe["mol"].notnull()].copy()
    return dataframe


def add_feature_scaffold(initial_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add Bemis-Murcko scaffold features to a molecular dataframe."""
    dataframe = initial_dataframe.copy()
    dataframe["murcko_scaffold"] = dataframe["mol"].apply(
        lambda m: MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None
    )
    dataframe["ring_count"] = dataframe["mol"].apply(
        lambda m: rdMolDescriptors.CalcNumRings(m) if m else None
    )
    dataframe["aromatic_ring_count"] = dataframe["mol"].apply(
        lambda m: rdMolDescriptors.CalcNumAromaticRings(m) if m else None
    )
    return dataframe


def add_rdkit_descriptor_features(initial_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add a compact, useful descriptor set for feature-store use."""
    dataframe = initial_dataframe.copy()
    dataframe["mol_weight"] = dataframe["mol"].apply(Descriptors.MolWt)
    dataframe["logp"] = dataframe["mol"].apply(Crippen.MolLogP)
    dataframe["tpsa"] = dataframe["mol"].apply(Descriptors.TPSA)
    dataframe["hba"] = dataframe["mol"].apply(Lipinski.NumHAcceptors)
    dataframe["hbd"] = dataframe["mol"].apply(Lipinski.NumHDonors)
    dataframe["rotatable_bonds"] = dataframe["mol"].apply(Lipinski.NumRotatableBonds)
    dataframe["heavy_atom_count"] = dataframe["mol"].apply(Lipinski.HeavyAtomCount)
    dataframe["fraction_csp3"] = dataframe["mol"].apply(Lipinski.FractionCSP3)
    return dataframe


def add_morgan_fingerprint_columns(
    initial_dataframe: pd.DataFrame,
    radius: int = 2,
    n_bits: int = 256,
    prefix: str = "mfp",
) -> pd.DataFrame:
    """Expand Morgan fingerprint bits into explicit feature columns."""
    dataframe = initial_dataframe.copy()

    def _fp_bits(mol: Chem.Mol) -> list[int]:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        return list(fp)

    bit_vectors = dataframe["mol"].apply(_fp_bits)
    fp_frame = pd.DataFrame(bit_vectors.tolist(), index=dataframe.index)
    fp_frame.columns = [f"{prefix}_{idx}" for idx in range(n_bits)]

    return pd.concat([dataframe, fp_frame], axis=1)


def build_feature_store_frame(
    query: str = "SELECT id, structure FROM molecules WHERE structure@>'c1cccnc1' LIMIT 100;",
    include_fingerprints: bool = False,
    fingerprint_bits: int = 256,
) -> pd.DataFrame:
    """End-to-end feature dataframe builder from cartridge query to store-ready rows."""
    feature_frame = datamol_clean_cartridge(query=query)
    if feature_frame.empty:
        return feature_frame

    feature_frame = add_rdkit_descriptor_features(feature_frame)
    feature_frame = add_feature_scaffold(feature_frame)

    if include_fingerprints:
        feature_frame = add_morgan_fingerprint_columns(
            feature_frame,
            n_bits=fingerprint_bits,
        )

    feature_frame["event_timestamp"] = pd.Timestamp.now(tz="UTC")
    return feature_frame


def filter_scaffolds(
    initial_dataframe: pd.DataFrame,
    min_ring_count: int = 1,
    min_aromatic_rings: int = 0,
) -> pd.DataFrame:
    """Filter molecules to richer scaffolds for downstream modeling."""
    return initial_dataframe[
        (initial_dataframe["ring_count"] >= min_ring_count)
        & (initial_dataframe["aromatic_ring_count"] >= min_aromatic_rings)
    ].copy()


@dataclass
class EMolculeConnect:
    smiles_column: str
    data: pd.DataFrame = field(default_factory=dm.data.freesolv)

    def _preprocess(self, row):
        mol = dm.to_mol(row[self.smiles_column], ordered=True)
        mol = dm.fix_mol(mol)
        mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
        mol = dm.standardize_mol(
            mol,
            disconnect_metals=False,
            normalize=True,
            reionize=True,
            uncharge=False,
            stereo=True,
        )

        row["standard_smiles"] = dm.standardize_smiles(dm.to_smiles(mol))
        row["selfies"] = dm.to_selfies(mol)
        row["inchi"] = dm.to_inchi(mol)
        row["inchikey"] = dm.to_inchikey(mol)
        return row


@dataclass
class Position:
    name: str
    lon: float = 0.0
    lat: float = 0.0

    def distance_to(self, other):
        """Compute great-circle distance in kilometers."""
        r = 6371
        lam_1, lam_2 = radians(self.lon), radians(other.lon)
        phi_1, phi_2 = radians(self.lat), radians(other.lat)
        h = (sin((phi_2 - phi_1) / 2)) ** 2 + cos(phi_1) * cos(phi_2) * sin(
            (lam_2 - lam_1) / 2
        ) ** 2
        return 2 * r * asin(sqrt(h))


if __name__ == "__main__":
    pandas_result = build_feature_store_frame(include_fingerprints=False)
    print(pandas_result.head())
