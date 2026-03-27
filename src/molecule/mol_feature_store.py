"""Feature-store export utilities for molecular cartridge data."""

from pathlib import Path

import pandas as pd

from mol_features import build_feature_store_frame


def export_feature_store_csv(
    output_path: str = "../../data/molecule_features.csv",
    include_fingerprints: bool = False,
    fingerprint_bits: int = 256,
) -> Path:
    """Build feature rows and persist them as a CSV for feature-store ingestion."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    molecular_data_source = build_feature_store_frame(
        include_fingerprints=include_fingerprints,
        fingerprint_bits=fingerprint_bits,
    )

    if molecular_data_source.empty:
        # Still produce a predictable file so downstream pipelines can branch gracefully.
        pd.DataFrame(columns=["id", "event_timestamp"]).to_csv(output, index=False)
        return output

    molecular_data_source.to_csv(output, index=False)
    return output


if __name__ == "__main__":
    output = export_feature_store_csv()
    print(f"wrote feature store export to: {output}")
