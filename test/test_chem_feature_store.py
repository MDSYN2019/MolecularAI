import csv

import pytest

from molecule.chem_feature_store.ingest import ingest_csv
from molecule.chem_feature_store.repository import MoleculeRepository


class RecordingRepository:
    def __init__(self):
        self.rows = []

    def upsert(self, smiles, name=None):
        self.rows.append((smiles, name))


def test_ingest_csv(tmp_path):
    csv_path = tmp_path / "molecules.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", "smiles"])
        writer.writeheader()
        writer.writerow({"name": "water", "smiles": "O"})
    repository = RecordingRepository()

    assert ingest_csv(csv_path, repository) == 1
    assert repository.rows == [("O", "water")]


def test_ingest_requires_smiles_column(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("name\nwater\n")

    with pytest.raises(ValueError, match="smiles"):
        ingest_csv(csv_path, RecordingRepository())


def test_similarity_threshold_is_validated_without_database():
    repository = MoleculeRepository(connection=None)

    with pytest.raises(ValueError, match="between 0 and 1"):
        repository.similar("CCO", threshold=1.1)
