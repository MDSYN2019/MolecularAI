CREATE EXTENSION IF NOT EXISTS rdkit;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS molecules (
    molecule_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name text,
    canonical_smiles text NOT NULL UNIQUE,
    structure mol NOT NULL,
    morgan_bfp bfp GENERATED ALWAYS AS
        (morganbv_fp(structure, 2, 2048)) STORED,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS molecules_structure_gist
    ON molecules USING gist (structure);
CREATE INDEX IF NOT EXISTS molecules_morgan_gist
    ON molecules USING gist (morgan_bfp);

CREATE TABLE IF NOT EXISTS molecule_features (
    molecule_id uuid REFERENCES molecules(molecule_id) ON DELETE CASCADE,
    canonical_smiles text NOT NULL,
    molecular_weight double precision NOT NULL,
    logp double precision NOT NULL,
    h_bond_donors bigint NOT NULL,
    h_bond_acceptors bigint NOT NULL,
    rotatable_bonds bigint NOT NULL,
    ring_count bigint NOT NULL,
    tpsa double precision NOT NULL,
    event_timestamp timestamptz NOT NULL,
    created_timestamp timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (molecule_id, event_timestamp)
);

CREATE INDEX IF NOT EXISTS molecule_features_event_time
    ON molecule_features (event_timestamp);
