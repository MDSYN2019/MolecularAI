"""Abstract base classes for molecular modelling workflows."""

from abc import ABC, abstractmethod


class MolecularModelling(ABC):
    """Common interface for modelling implementations."""

    @abstractmethod
    def placeholder(self):
        """Provide a minimal overridable hook for subclasses."""
        raise NotImplementedError
