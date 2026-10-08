# ML-агент для табличных данных

ИИ-агент, который сам строит и улучшает ML-модели для CSV-таблиц
(классификация и регрессия). Проект для конкурса Альфа-Банка.

> Сейчас готов только каркас проекта. Модули будут реализованы по этапам.

## Установка на Windows

### 1. Установите Python 3.12

Нужен именно **Python 3.12**. На Python 3.14 библиотека LightAutoML не устанавливается.

Скачайте установщик «Windows installer (64-bit)» для 3.12.x с
https://www.python.org/downloads/windows/ и при установке оставьте галочку **py launcher**.

Проверка (в PowerShell):

```powershell
py -0p
```

В списке должна быть строка с `3.12`.

### 2. Скачайте проект

```powershell
git clone <ссылка-на-репозиторий>
cd MLII
```

### 3. Создайте виртуальное окружение и установите зависимости

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Установка занимает несколько минут: LightAutoML тянет за собой PyTorch, CatBoost и др.

Если `Activate.ps1` выдаёт ошибку про «выполнение сценариев отключено», выполните один раз:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 4. Настройте ключи

```powershell
copy .env.example .env
```

Откройте `.env` и впишите свои API-ключи. Этот файл не попадает в git.

### 5. Подготовьте учебные датасеты

```powershell
python scripts\prepare_datasets.py
```

Скрипт скачает Titanic и California Housing и сгенерирует датасет с ловушками в папку `data/`.

### 6. Проверьте, что всё работает

```powershell
python -m pytest
```

## Структура проекта

```
src/            код агента (7 модулей)
data/           датасеты CSV (не в git)
experiments/    результаты и отчёты (не в git)
tests/          тесты
scripts/        вспомогательные скрипты (подготовка данных)
```

## Анализ датасета

```powershell
python -m src.data_profiler data\titanic.csv Survived
```

Подробное описание архитектуры — в [CLAUDE.md](CLAUDE.md).
