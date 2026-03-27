from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_PROPERTIES = ["Tg", "FFV", "Tc", "Density", "Rg"]


@dataclass
class PolymerGNNConfig:
    """
    Lightweight, file-based MLOps config for the polymer GNN workflow.
    """

    experiment_name: str = "polymer-gnn-baseline"
    seed: int = 42
    train_csv: str = "data/raw/train.csv"
    test_csv: str = "data/raw/test.csv"
    properties: list[str] = field(default_factory=lambda: DEFAULT_PROPERTIES.copy())
    batch_size: int = 64
    hidden_channels: int = 384
    num_layers: int = 8
    dropout: float = 0.25
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    eval_every: int = 5

    @classmethod
    def from_json(cls, config_path: str | Path) -> "PolymerGNNConfig":
        payload = json.loads(Path(config_path).read_text())
        return cls(**payload)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


class ExperimentTracker:
    """
    Minimal run tracker that stores params and metrics under a run directory.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"

    def log_params(self, params: dict[str, Any]) -> None:
        payload = {"timestamp": utc_now(), "params": params}
        (self.run_dir / "params.json").write_text(json.dumps(payload, indent=2) + "\n")

    def log_metric(self, step: int, **metrics: float) -> None:
        payload = {
            "timestamp": utc_now(),
            "step": step,
            "metrics": metrics,
        }
        with self.metrics_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload) + "\n")

    def log_artifact(self, src: str | Path, artifact_name: str | None = None) -> Path:
        src_path = Path(src)
        dest_name = artifact_name if artifact_name else src_path.name
        dest_path = self.run_dir / "artifacts" / dest_name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(src_path.read_bytes())
        return dest_path


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def bootstrap_mlops_template(root_dir: str | Path) -> dict[str, Path]:
    """
    Create a reusable folder template around the existing polymer GNN project.
    """

    root = Path(root_dir)
    paths = {
        "root": root,
        "configs": root / "configs",
        "data_raw": root / "data" / "raw",
        "data_processed": root / "data" / "processed",
        "artifacts_models": root / "artifacts" / "models",
        "artifacts_reports": root / "artifacts" / "reports",
        "runs": root / "runs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    base_config = PolymerGNNConfig()
    config_path = paths["configs"] / "polymer_gnn.base.json"
    if not config_path.exists():
        base_config.to_json(config_path)

    readme_path = root / "README.md"
    if not readme_path.exists():
        readme_path.write_text(
            "\n".join(
                [
                    "# Polymer GNN MLOps Template",
                    "",
                    "This scaffold stores config, data, artifacts, and run logs in a",
                    "reproducible directory layout.",
                    "",
                    "## Quick start",
                    "1. Edit `configs/polymer_gnn.base.json`.",
                    "2. Place train/test CSV files under `data/raw/`.",
                    "3. Run your training entrypoint and log metrics into `runs/`.",
                    "",
                ]
            )
            + "\n"
        )

    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap polymer GNN MLOps template.")
    parser.add_argument(
        "--root-dir",
        default="polymer_gnn_mlops",
        help="Directory where the MLOps template will be created.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = bootstrap_mlops_template(args.root_dir)
    print(f"Template created at: {paths['root']}")
    print(f"Base config: {paths['configs'] / 'polymer_gnn.base.json'}")


if __name__ == "__main__":
    main()
