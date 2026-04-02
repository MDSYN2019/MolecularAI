from graphein.protein.config import ProteinGraphConfig

def build_default_config() -> ProteinGraphConfig:
    """Return the default Graphein protein-graph configuration."""
    return ProteinGraphConfig()


if __name__ == "__main__":
    # Keep demo output only when running as script (not on import).
    print(build_default_config())
