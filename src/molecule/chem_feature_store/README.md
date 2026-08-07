# RDKit molecule library and Feast feature store

This directory provides a reproducible molecular feature platform:

* PostgreSQL with the RDKit cartridge is the system of record for structures,
  fingerprints, and calculated descriptors.
* Feast reads descriptor history from PostgreSQL for point-in-time-correct
  training sets and materializes it to a local online store.
* `MoleculeRepository` provides ingestion, exact lookup, substructure search,
  and Tanimoto similarity search without hiding SQL behind an ORM.

## Start locally

Docker and Python 3.10+ are required.

```bash
cd src/molecule/chem_feature_store
cp .env.example .env
docker compose up -d --wait
python -m pip install -r requirements.txt
set -a && source .env && set +a
python -m chem_feature_store.ingest examples/molecules.csv

cd feature_repo
feast apply
feast materialize-incremental "$(date -u +%Y-%m-%dT%H:%M:%S)"
```

Run the commands from `src/molecule` (or install the root project) so that the
`chem_feature_store` package is importable. The example CSV requires a
`smiles` column and may include a `name` column.

## Retrieve features

```python
from datetime import datetime, timezone
import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")
entity_df = pd.DataFrame({
    "molecule_id": ["00000000-0000-0000-0000-000000000000"],
    "event_timestamp": [datetime.now(timezone.utc)],
})
training = store.get_historical_features(
    entity_df,
    features=["molecular_descriptors:molecular_weight",
              "molecular_descriptors:logp"],
).to_df()
```

For production, replace the local Feast registry and SQLite online store with
shared services, keep credentials in a secret manager, and run descriptor
ingestion and materialization as scheduled jobs. PostgreSQL remains the
offline source and searchable structure library.

## Database behavior

The schema deduplicates canonical SMILES. It stores an RDKit `mol` value
and a generated Morgan fingerprint, and creates GiST indexes for substructure
and similarity operators. Re-ingesting a structure updates its metadata and
appends a time-stamped descriptor record for Feast. Invalid SMILES are rejected
before a transaction starts.

```bash
pytest -q
docker compose down -v
```
