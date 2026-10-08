"""Тесты профайлера данных (src/data_profiler.py).

Тесты не зависят от файлов в data/: нужные таблицы создаются прямо здесь.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_profiler import (
    detect_column_type,
    detect_task_type,
    profile_dataframe,
    profile_dataset,
    profile_to_text,
)


def make_tricky_dataframe(n_rows: int = 500) -> pd.DataFrame:
    """Таблица с ловушками: ID, константа и утечка (копия цели с шумом)."""
    rng = np.random.default_rng(0)
    target = rng.integers(0, 2, n_rows)
    leak = target.copy()
    leak[:5] = 1 - leak[:5]  # чуть-чуть «испортим», чтобы не была точной копией
    return pd.DataFrame({
        "row_number": np.arange(n_rows),
        "noise": rng.normal(size=n_rows),
        "color": rng.choice(["red", "green", "blue"], n_rows),
        "constant": "same",
        "leak": leak,
        "target": target,
    })


# --- Тип задачи ---

@pytest.mark.parametrize("values, expected", [
    ([0, 1, 1, 0, 1], "binary"),
    (["yes", "no", "no", "yes"], "binary"),
    ([0, 1, 2, 3, 2, 1], "multiclass"),
    (["cat", "dog", "mouse", "cat"], "multiclass"),
    (list(np.linspace(0, 100, 50)), "regression"),
])
def test_detect_task_type(values, expected):
    assert detect_task_type(pd.Series(values)) == expected


def test_task_type_ignores_missing_values():
    assert detect_task_type(pd.Series([0, 1, None, 1, 0])) == "binary"


# --- Тип столбца ---

def test_detect_column_type():
    assert detect_column_type(pd.Series([1.5, 2.0, 3.1])) == "number"
    assert detect_column_type(pd.Series(["a", "b", "a"])) == "category"
    assert detect_column_type(pd.Series(["2023-01-01", "2023-02-15", None])) == "date"
    assert detect_column_type(pd.Series(["31.12.2022", "01.01.2023"])) == "date"
    texts = pd.Series([f"очень длинный отзыв номер {i}" for i in range(10)])
    assert detect_column_type(texts) == "text"


# --- Ловушки ---

def test_finds_id_constant_and_leakage():
    profile = profile_dataframe(make_tricky_dataframe(), "target")

    assert list(profile["id_columns"]) == ["row_number"]
    assert profile["constant_columns"] == ["constant"]
    assert list(profile["leakage_columns"]) == ["leak"]


def test_normal_columns_are_not_suspicious():
    profile = profile_dataframe(make_tricky_dataframe(), "target")
    suspicious = (
        set(profile["id_columns"]) | set(profile["constant_columns"]) | set(profile["leakage_columns"])
    )
    assert "noise" not in suspicious
    assert "color" not in suspicious


def test_id_detected_by_name():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "user_id": rng.permutation(10_000)[:100] * 7,  # разные, но не подряд
        "target": rng.normal(size=100),
    })
    assert "user_id" in profile_dataframe(df, "target")["id_columns"]


def test_leakage_in_regression():
    rng = np.random.default_rng(2)
    target = rng.normal(size=400)
    df = pd.DataFrame({
        "almost_target": target * 2 + rng.normal(scale=0.01, size=400),
        "noise": rng.normal(size=400),
        "target": target,
    })
    profile = profile_dataframe(df, "target")
    assert profile["target"]["task_type"] == "regression"
    assert list(profile["leakage_columns"]) == ["almost_target"]


# --- Общий результат ---

def test_profile_dataset_reads_csv(tmp_path):
    csv_path = tmp_path / "data.csv"
    make_tricky_dataframe().to_csv(csv_path, index=False)

    profile = profile_dataset(str(csv_path), "target")

    assert profile["file"] == "data.csv"
    assert profile["n_rows"] == 500
    assert profile["n_columns"] == 6
    assert profile["target"]["task_type"] == "binary"
    assert set(profile["columns"]) == {"row_number", "noise", "color", "constant", "leak"}
    assert profile["columns"]["color"]["type"] == "category"
    assert profile["columns"]["noise"]["missing_share"] == 0


def test_missing_target_column_raises_error():
    with pytest.raises(ValueError, match="не найден"):
        profile_dataframe(make_tricky_dataframe(), "no_such_column")


def test_text_mentions_all_problems():
    text = profile_to_text(profile_dataframe(make_tricky_dataframe(), "target"))

    assert "бинарная классификация" in text
    assert "row_number: похоже на ID" in text
    assert "constant: константа" in text
    assert "leak: возможная утечка" in text
