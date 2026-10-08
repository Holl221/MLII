"""Модуль 1. Анализ датасета (профайлер).

Что делает:
- загружает CSV и определяет тип задачи по целевому столбцу:
  бинарная классификация, многоклассовая классификация или регрессия;
- для каждого столбца находит тип (число, категория, дата, текст),
  долю пропусков и число уникальных значений;
- ищет подозрительное: ID-столбцы, константные столбцы и возможную
  утечку данных (столбец слишком хорошо предсказывает цель);
- умеет превращать результат в короткий текст на русском для LLM.

Запуск из командной строки:
    .\\.venv\\Scripts\\python.exe -m src.data_profiler data\\titanic.csv Survived
"""

import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# --- Настройки (пороги) ---

# Целое число с таким или меньшим числом разных значений считаем классами
MAX_CLASSES = 20
# Строковый столбец считаем текстом, если в среднем в нём столько слов или больше...
TEXT_MIN_WORDS = 3
# ...и при этом большинство значений разные
TEXT_MIN_UNIQUE_SHARE = 0.5
# Доля значений, похожих на дату, чтобы назвать столбец датой
DATE_MIN_SHARE = 0.9
# Если один столбец предсказывает цель с таким качеством или выше — подозреваем утечку
LEAKAGE_THRESHOLD = 0.95
# Для проверки утечки берём не больше стольких строк (чтобы было быстро)
LEAKAGE_MAX_ROWS = 20_000

# Похоже на дату: 2023-01-31, 2023/1/31, 31.01.2023, 31/01/2023
DATE_PATTERN = re.compile(r"^(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4})")

TASK_NAMES = {
    "binary": "бинарная классификация",
    "multiclass": "многоклассовая классификация",
    "regression": "регрессия",
}
TYPE_NAMES = {"number": "число", "category": "категория", "date": "дата", "text": "текст"}


# --- Тип задачи ---

def detect_task_type(target: pd.Series) -> str:
    """Определить тип задачи по целевому столбцу: binary, multiclass или regression."""
    values = target.dropna()
    n_unique = values.nunique()

    if n_unique <= 2:
        return "binary"
    # Строки (например, "кошка"/"собака"/"мышь") — это всегда классы
    if not pd.api.types.is_numeric_dtype(values):
        return "multiclass"
    # Немного разных целых чисел (0, 1, 2, ...) — тоже классы
    is_integer = (values % 1 == 0).all()
    if is_integer and n_unique <= MAX_CLASSES:
        return "multiclass"
    return "regression"


# --- Тип столбца ---

def detect_column_type(column: pd.Series) -> str:
    """Определить тип столбца: number, category, date или text."""
    if pd.api.types.is_datetime64_any_dtype(column):
        return "date"
    if pd.api.types.is_numeric_dtype(column):
        return "number"

    values = column.dropna().astype(str)
    if len(values) == 0:
        return "category"

    # Дата: почти все значения выглядят как дата
    sample = values.head(1000)
    if sample.str.match(DATE_PATTERN).mean() >= DATE_MIN_SHARE:
        return "date"

    # Текст: длинные фразы из нескольких слов, почти все разные
    avg_words = values.str.split().str.len().mean()
    unique_share = values.nunique() / len(values)
    if avg_words >= TEXT_MIN_WORDS and unique_share >= TEXT_MIN_UNIQUE_SHARE:
        return "text"

    return "category"


# --- Подозрительные столбцы ---

def name_looks_like_id(name: str) -> bool:
    """Похоже ли название столбца на идентификатор (id, user_id, PassengerId, ...)."""
    lower = name.lower()
    return (
        lower in ("id", "index", "idx")
        or lower.endswith("_id")
        or name.endswith("Id")  # PassengerId, userId
        or name.endswith("ID")  # clientID
        or lower.startswith("unnamed")  # лишний индекс, сохранённый pandas
    )


def find_id_reason(column: pd.Series, column_type: str) -> str | None:
    """Вернуть причину, если столбец похож на ID, иначе None."""
    values = column.dropna()
    if len(values) < 20:
        return None  # слишком мало строк, чтобы судить
    unique_share = values.nunique() / len(values)

    # Целые числа, все разные и идут подряд без пропусков: 1, 2, 3, ...
    if column_type == "number" and (values % 1 == 0).all() and unique_share == 1:
        if values.max() - values.min() + 1 == len(values):
            return "все значения разные и идут подряд, как номер строки"

    if name_looks_like_id(str(column.name)) and unique_share >= 0.9:
        return "название похоже на ID и почти все значения разные"

    if column_type == "category" and unique_share == 1:
        return "все строковые значения разные, похоже на код"

    return None


def is_constant(column: pd.Series) -> bool:
    """Столбец константный, если в нём одно значение (или только пропуски)."""
    return column.nunique(dropna=False) <= 1


def single_feature_score(feature: pd.Series, target: pd.Series, task_type: str) -> float:
    """Насколько хорошо ОДИН столбец предсказывает цель (оценка от «плохо» до 1.0).

    Обучаем маленькое дерево решений только на этом столбце и проверяем его
    на кросс-валидации (на строках, которых дерево не видело).
    Классификация — сбалансированная точность (1.0 = всегда угадывает).
    Регрессия — R² (1.0 = идеальный прогноз, 0 = не лучше среднего).
    """
    if pd.api.types.is_numeric_dtype(feature):
        x = feature.astype(float)
    else:
        # Категории превращаем в номера: "male" -> 0, "female" -> 1, пропуск -> -1
        x = feature.astype("category").cat.codes
    x = x.to_frame()

    if task_type == "regression":
        model = DecisionTreeRegressor(min_samples_leaf=20, random_state=0)
        folds = KFold(n_splits=3, shuffle=True, random_state=0)
        predictions = cross_val_predict(model, x, target, cv=folds)
        return float(r2_score(target, predictions))

    model = DecisionTreeClassifier(min_samples_leaf=20, random_state=0)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    predictions = cross_val_predict(model, x, target, cv=folds)
    return float(balanced_accuracy_score(target, predictions))


# --- Сборка профиля ---

def describe_target(target: pd.Series, task_type: str) -> dict:
    """Краткое описание целевого столбца."""
    info = {
        "task_type": task_type,
        "missing": int(target.isna().sum()),
        "n_unique": int(target.nunique()),
    }
    if task_type == "regression":
        info["min"] = float(target.min())
        info["median"] = float(target.median())
        info["max"] = float(target.max())
    else:
        # Доли классов: {класс: доля}, самые частые первыми (не больше 10)
        shares = target.value_counts(normalize=True).head(10)
        info["class_shares"] = {str(k): round(float(v), 3) for k, v in shares.items()}
    return info


def profile_dataframe(df: pd.DataFrame, target: str) -> dict:
    """Построить профиль таблицы. Возвращает словарь (см. profile_dataset)."""
    if target not in df.columns:
        raise ValueError(f"Целевой столбец '{target}' не найден. Есть столбцы: {list(df.columns)}")

    task_type = detect_task_type(df[target])

    # Для проверки утечки нужны строки с известной целью (и не слишком много)
    known = df[df[target].notna()]
    if len(known) > LEAKAGE_MAX_ROWS:
        known = known.sample(LEAKAGE_MAX_ROWS, random_state=0)

    columns = {}
    id_columns = {}
    constant_columns = []
    leakage_columns = {}

    for name in df.columns:
        if name == target:
            continue
        column = df[name]
        column_type = detect_column_type(column)
        info = {
            "type": column_type,
            "missing_share": round(float(column.isna().mean()), 3),
            "n_unique": int(column.nunique()),
            "score": None,  # сила связи с целью (если считали)
        }

        id_reason = find_id_reason(column, column_type)
        if is_constant(column):
            constant_columns.append(name)
        elif id_reason:
            id_columns[name] = id_reason
        elif column_type in ("number", "category"):
            # Даты и тексты деревом напрямую не проверить — пропускаем
            score = single_feature_score(known[name], known[target], task_type)
            info["score"] = round(score, 3)
            if score >= LEAKAGE_THRESHOLD:
                leakage_columns[name] = round(score, 3)

        columns[name] = info

    return {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "target_name": target,
        "target": describe_target(df[target], task_type),
        "columns": columns,
        "id_columns": id_columns,
        "constant_columns": constant_columns,
        "leakage_columns": leakage_columns,
    }


def profile_dataset(csv_path: str, target: str) -> dict:
    """Загрузить CSV и вернуть словарь с описанием датасета (профиль).

    Ключи словаря:
    - file, n_rows, n_columns, target_name;
    - target: тип задачи, пропуски, доли классов или min/median/max;
    - columns: {столбец: {type, missing_share, n_unique, score}};
    - id_columns: {столбец: причина};
    - constant_columns: [столбцы];
    - leakage_columns: {столбец: оценка}.
    """
    df = pd.read_csv(csv_path)
    profile = profile_dataframe(df, target)
    profile["file"] = Path(csv_path).name
    return profile


# --- Текст для LLM ---

def format_percent(share: float) -> str:
    """0.123 -> '12%'; маленькие ненулевые доли не превращаем в 0%."""
    if 0 < share < 0.01:
        return "<1%"
    return f"{share:.0%}"


def profile_to_text(profile: dict) -> str:
    """Превратить профиль в короткий понятный текст на русском."""
    target = profile["target"]
    task_type = target["task_type"]
    lines = []

    file_name = profile.get("file", "таблица")
    lines.append(f"Датасет {file_name}: {profile['n_rows']} строк, {profile['n_columns']} столбцов.")

    lines.append(f"Цель: {profile['target_name']} — {TASK_NAMES[task_type]}.")
    if task_type == "regression":
        lines.append(f"Значения цели: от {target['min']:.4g} до {target['max']:.4g}, медиана {target['median']:.4g}.")
    else:
        shares = ", ".join(f"{k}: {format_percent(v)}" for k, v in target["class_shares"].items())
        lines.append(f"Классы ({target['n_unique']}): {shares}.")
    if target["missing"]:
        lines.append(f"Пропусков в цели: {target['missing']} (такие строки нельзя использовать для обучения).")

    metric = "R²" if task_type == "regression" else "сбаланс. точность"
    lines.append("")
    lines.append(f"Признаки (тип; пропуски; уникальных; сила — {metric} прогноза цели по одному столбцу):")
    for name, info in profile["columns"].items():
        parts = [TYPE_NAMES[info["type"]], f"пропуски {format_percent(info['missing_share'])}", f"уник. {info['n_unique']}"]
        if info["score"] is not None:
            # "+ 0.0" убирает некрасивое "-0.00" у чисел вроде -0.001
            parts.append(f"сила {round(info['score'], 2) + 0.0:.2f}")
        lines.append(f"- {name}: " + "; ".join(parts))

    lines.append("")
    lines.append("Подозрительное:")
    problems = []
    for name, reason in profile["id_columns"].items():
        problems.append(f"- {name}: похоже на ID ({reason}) — удалить из признаков.")
    for name in profile["constant_columns"]:
        problems.append(f"- {name}: константа (одно значение) — удалить.")
    for name, score in profile["leakage_columns"].items():
        problems.append(
            f"- {name}: возможная утечка (один столбец даёт {metric} {score:.2f}) — "
            "проверить, известен ли он до момента прогноза."
        )
    lines.extend(problems or ["- ничего не найдено."])

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Использование: python -m src.data_profiler <путь к CSV> <целевой столбец>")
        sys.exit(1)
    print(profile_to_text(profile_dataset(sys.argv[1], sys.argv[2])))
