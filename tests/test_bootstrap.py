"""Bootstrap smoke test — verifies the package can be imported and version is set."""


def test_version_string() -> None:
    from engine import __version__

    assert __version__ == "0.1.0"
