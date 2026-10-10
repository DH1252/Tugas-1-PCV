import pytest

from vision.config import load_config


@pytest.fixture
def cfg():
    return load_config()
