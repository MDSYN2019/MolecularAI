"""Optional WeightWatcher integration helpers."""


def load_weightwatcher():
    """Return the ``weightwatcher`` module if installed.

    Raises:
        ImportError: if ``weightwatcher`` is not installed.
    """

    import weightwatcher as ww

    return ww
