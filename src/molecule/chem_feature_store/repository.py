"""PostgreSQL-backed RDKit molecule library."""

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Any


@dataclass(frozen=True)
class MolecularFeatures:
    canonical_smiles: str
    molecular_weight: float
    logp: float
    h_bond_donors: int
    h_bond_acceptors: int
    rotatable_bonds: int
    ring_count: int
    tpsa: float


def calculate_features(smiles: str) -> MolecularFeatures:
    """Validate a SMILES string and calculate deterministic RDKit descriptors."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    return MolecularFeatures(
        canonical_smiles=canonical,
        molecular_weight=float(Descriptors.MolWt(molecule)),
        logp=float(Descriptors.MolLogP(molecule)),
        h_bond_donors=int(Lipinski.NumHDonors(molecule)),
        h_bond_acceptors=int(Lipinski.NumHAcceptors(molecule)),
        rotatable_bonds=int(Lipinski.NumRotatableBonds(molecule)),
        ring_count=int(Lipinski.RingCount(molecule)),
        tpsa=float(rdMolDescriptors.CalcTPSA(molecule)),
    )


class MoleculeRepository:
    """Persist descriptors and use cartridge operators for chemical search."""

    def __init__(self, connection: Any):
        self.connection = connection

    @classmethod
    def connect(cls, dsn: str | None = None) -> "MoleculeRepository":
        import psycopg

        dsn = dsn or os.getenv(
            "MOLECULE_DATABASE_URL",
            "postgresql://molecule:molecule@localhost:5432/molecules",
        )
        return cls(psycopg.connect(dsn))

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "MoleculeRepository":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def upsert(self, smiles: str, name: str | None = None) -> str:
        features = calculate_features(smiles)
        observed_at = datetime.now(timezone.utc)
        with self.connection.transaction(), self.connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO molecules (name, canonical_smiles, structure)
                   VALUES (%s, %s, mol_from_smiles(%s))
                   ON CONFLICT (canonical_smiles) DO UPDATE
                   SET name = COALESCE(EXCLUDED.name, molecules.name),
                       updated_at = now()
                   RETURNING molecule_id""",
                (name, features.canonical_smiles, features.canonical_smiles),
            )
            molecule_id = cursor.fetchone()[0]
            cursor.execute(
                """INSERT INTO molecule_features
                       (molecule_id, canonical_smiles, molecular_weight, logp,
                        h_bond_donors, h_bond_acceptors, rotatable_bonds,
                        ring_count, tpsa, event_timestamp)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   """,
                (molecule_id, *features.__dict__.values(), observed_at),
            )
        return str(molecule_id)

    def substructure(self, query_smiles: str, limit: int = 100) -> list[dict]:
        return self._search(
            "structure @> mol_from_smiles(%s)", (query_smiles, limit)
        )

    def similar(
        self, query_smiles: str, threshold: float = 0.7, limit: int = 100
    ) -> list[dict]:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        fingerprint = "morganbv_fp(mol_from_smiles(%s), 2, 2048)"
        sql = f"""SELECT molecule_id::text, name, canonical_smiles,
                          tanimoto_sml(morgan_bfp, {fingerprint}) AS similarity
                   FROM molecules
                   WHERE tanimoto_sml(morgan_bfp, {fingerprint}) >= %s
                   ORDER BY similarity DESC LIMIT %s"""
        with self.connection.cursor() as cursor:
            cursor.execute(sql, (query_smiles, query_smiles, threshold, limit))
            return self._rows(cursor)

    def _search(self, predicate: str, parameters: tuple) -> list[dict]:
        sql = f"""SELECT molecule_id::text, name, canonical_smiles
                   FROM molecules WHERE {predicate}
                   ORDER BY canonical_smiles LIMIT %s"""
        with self.connection.cursor() as cursor:
            cursor.execute(sql, parameters)
            return self._rows(cursor)

    @staticmethod
    def _rows(cursor: Any) -> list[dict]:
        columns = [column.name for column in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
