"""CSV ingestion command for the molecule library."""

import argparse
import csv
from pathlib import Path

from .repository import MoleculeRepository


def ingest_csv(path: Path, repository: MoleculeRepository) -> int:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "smiles" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'smiles' column")
        count = 0
        for line_number, row in enumerate(reader, start=2):
            try:
                repository.upsert(row["smiles"], row.get("name") or None)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            count += 1
        return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--dsn", help="PostgreSQL DSN (or MOLECULE_DATABASE_URL)")
    args = parser.parse_args()
    with MoleculeRepository.connect(args.dsn) as repository:
        count = ingest_csv(args.csv_file, repository)
    print(f"Ingested {count} molecules")


if __name__ == "__main__":
    main()
