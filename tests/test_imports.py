"""Простейший тест: все модули каркаса импортируются без ошибок."""

import importlib

import pytest

MODULES = [
    "src.data_profiler",
    "src.baseline",
    "src.llm_client",
    "src.agent",
    "src.sandbox",
    "src.validation",
    "src.report",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)
