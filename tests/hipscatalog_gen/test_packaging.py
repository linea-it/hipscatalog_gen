import hipscatalog_gen


def test_version():
    """Check to see that we can get the package version"""
    assert hipscatalog_gen.__version__ is not None
