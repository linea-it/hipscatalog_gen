"""Packaging sanity checks for hipscatalog_gen (installed metadata)."""

import hipscatalog_gen


def test_version():
    """Check to see that we can get the package version"""
    assert hipscatalog_gen.__version__ is not None


"""Packaging sanity checks for hipscatalog_gen (installed metadata)."""
