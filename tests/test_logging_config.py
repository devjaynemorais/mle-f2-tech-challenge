"""Testes da configuração central de logging (src/utils/logging_config.py)."""

import logging

from src.utils.logging_config import setup_logging


def test_setup_logging_configures_root_logger_level_and_handler():
    setup_logging(level=logging.DEBUG)
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    handler = root.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.formatter is not None
    assert "%(levelname)" in handler.formatter._fmt
