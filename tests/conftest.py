"""Conftest for pytest configuration."""

import pytest
from loguru import logger


@pytest.fixture(autouse=True)
def log_test_name(request):
    logger.info(f"Running test: {request.node.name}")
