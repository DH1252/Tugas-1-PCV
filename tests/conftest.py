import pytest

from vision.config import load_config


@pytest.fixture
def cfg():
    c = load_config()
    c["vision"]["cursor_anchor"] = "palm"
    return c


@pytest.fixture
def cfg_top():
    return load_config()
