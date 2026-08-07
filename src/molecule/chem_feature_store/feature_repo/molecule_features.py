"""Feast definitions for molecular descriptors stored in PostgreSQL."""

from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Float64, Int64, String

molecule = Entity(
    name="molecule",
    join_keys=["molecule_id"],
    description="Stable identifier assigned by the RDKit molecule library",
)

molecule_features_source = PostgreSQLSource(
    name="molecule_features_source",
    query="""
        SELECT molecule_id::text AS molecule_id,
               canonical_smiles, molecular_weight, logp,
               h_bond_donors, h_bond_acceptors, rotatable_bonds,
               ring_count, tpsa, event_timestamp, created_timestamp
        FROM molecule_features
    """,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

molecular_descriptors = FeatureView(
    name="molecular_descriptors",
    entities=[molecule],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="canonical_smiles", dtype=String),
        Field(name="molecular_weight", dtype=Float64),
        Field(name="logp", dtype=Float64),
        Field(name="h_bond_donors", dtype=Int64),
        Field(name="h_bond_acceptors", dtype=Int64),
        Field(name="rotatable_bonds", dtype=Int64),
        Field(name="ring_count", dtype=Int64),
        Field(name="tpsa", dtype=Float64),
    ],
    source=molecule_features_source,
    online=True,
    tags={"domain": "cheminformatics", "generator": "rdkit"},
)

molecule_model_features = FeatureService(
    name="molecule_model_features_v1",
    features=[molecular_descriptors],
)
