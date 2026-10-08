"""Подготовка учебных датасетов в папке data/.

Папка data/ не попадает в git, поэтому датасеты создаются этим скриптом.
Запуск (из корня проекта):
    .\\.venv\\Scripts\\python.exe scripts\\prepare_datasets.py

Создаёт три файла:
1. titanic.csv   — бинарная классификация (цель: Survived)
2. housing.csv   — регрессия, California Housing (цель: MedHouseVal)
3. synthetic.csv — искусственный датасет с ловушками (цель: default)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Классический Titanic (тот же формат, что на Kaggle)
TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


def make_titanic() -> pd.DataFrame:
    """Скачать датасет Titanic."""
    return pd.read_csv(TITANIC_URL)


def make_housing() -> pd.DataFrame:
    """Загрузить California Housing из sklearn (признаки + цель в одной таблице)."""
    return fetch_california_housing(as_frame=True).frame


def make_synthetic(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Сгенерировать датасет «кредитный дефолт» со встроенными ловушками.

    Ловушки:
    - client_id        — идентификатор клиента (1, 2, 3, ...), бесполезен для прогноза;
    - country          — константа: у всех клиентов "RU";
    - collection_calls — утечка: звонки коллекторов бывают только ПОСЛЕ дефолта,
                         то есть столбец почти повторяет целевую переменную.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 70, n_rows)
    income = rng.normal(60_000, 20_000, n_rows).round(-2)
    loan_amount = rng.normal(300_000, 100_000, n_rows).round(-3)
    city = rng.choice(["Москва", "Казань", "Новосибирск", "Пермь"], n_rows)
    signup_date = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 700, n_rows), unit="D")

    # Честная зависимость: чем больше кредит относительно дохода, тем выше риск
    risk = loan_amount / income / 10 - 0.3 + rng.normal(0, 0.15, n_rows)
    default = (risk > 0.3).astype(int)

    # Утечка: у дефолтных клиентов 1–10 звонков, у остальных 0.
    # В 2% строк значение «перепутано», чтобы это была не точная копия цели.
    collection_calls = np.where(default == 1, rng.integers(1, 11, n_rows), 0)
    noise = rng.random(n_rows) < 0.02
    collection_calls[noise] = np.where(default[noise] == 1, 0, 1)

    return pd.DataFrame({
        "client_id": np.arange(1, n_rows + 1),
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "city": city,
        "signup_date": signup_date.strftime("%Y-%m-%d"),
        "country": "RU",
        "collection_calls": collection_calls,
        "default": default,
    })


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    datasets = {
        "titanic.csv": make_titanic,
        "housing.csv": make_housing,
        "synthetic.csv": make_synthetic,
    }
    for file_name, make in datasets.items():
        df = make()
        df.to_csv(DATA_DIR / file_name, index=False)
        print(f"{file_name}: {len(df)} строк, {df.shape[1]} столбцов")


if __name__ == "__main__":
    main()
