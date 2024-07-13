import pytest


def pytest_addoption(parser):
    """Shared command line options for pytest."""
    parser.addoption("--xpassthrough", default="false", action="store")


@pytest.fixture()
def do_debug_plot(pytestconfig):
    if pytestconfig.getoption("xpassthrough") == 'true':
        return True

    return False
