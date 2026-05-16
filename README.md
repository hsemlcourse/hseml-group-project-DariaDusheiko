# ML Project — Predicted Returns

**Студент:** Душейко Дарья Васильевна

**Группа:** БИВ231

## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Линтеры](#линтеры)
7. [Отчёт](#отчёт)

## Описание задачи

**Задача:** бинарная классификация возврата товара (`isReturned`), где `0` - не возврат, `1` - возврат.

**Датасет:** транзакции маркетплейса с OSF ([osfstorage](https://osf.io/c793h/files/osfstorage)); в проекте данные лежат локально в `.p`-файлах и архиве `data/processed/osfstorage-archive.zip`.

**Целевая метрика:** `ROC-AUC` (дополнительно в экспериментах считаются `F1`, `Precision`, `Recall`, `Accuracy`, `PR-AUC`).

## Структура репозитория

```text
.
├── .github/workflows/ci.yml
├── data
│   ├── processed
│   │   ├── customer_nodes_training.p
│   │   ├── customer_nodes_testing.p
│   │   ├── event_table_training.p
│   │   ├── event_table_testing.p
│   │   ├── product_nodes_training.p
│   │   ├── product_nodes_testing.p
│   │   └── osfstorage-archive.zip
│   └── raw
│       ├── customer_nodes_training.p
│       ├── customer_nodes_testing.p
│       ├── event_table_training.p
│       ├── event_table_testing.p
│       ├── product_nodes_training.p
│       └── product_nodes_testing.p
├── models
│   └── catboost_model.json
├── notebooks
│   ├── TABM.ipynb
│   ├── graphics.ipynb
│   ├── price_AB.ipynb
│   ├── сatboost_for_returns.ipynb
│   └── сatboost_test.ipynb
├── presentation
│   └── README.md
├── report
│   ├── calibration_curve.png
│   ├── confusion_matrix.png
│   ├── precision_recall.png
│   ├── proba_histogram.png
│   ├── roc_curve.png
│   └── report.md
├── src
├── tests
├── Dockerfile
├── requirements.txt
└── README.md
```

## Запуск

```bash
# 1. Клонировать репозиторий
git clone https://github.com/hsemlcourse/hseml-group-project-DariaDusheiko.git
cd hseml-group-project-DariaDusheiko

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt
```

Альтернативно можно запускать проект через Docker (в репозитории есть `Dockerfile`):

```bash
docker build -t returns-project .
docker run --rm -p 8888:8888 returns-project
```

## Данные

- `data/raw/` - исходные `.p`-файлы.
- `data/processed/` - подготовленные `.p`-файлы и архив `osfstorage-archive.zip`.
- По `report/report.md` используются таблицы событий (`event_table_*`) и узлы клиентов/товаров (`customer_nodes_*`, `product_nodes_*`) для train/test.

## Результаты

Результаты ниже приведены только из зафиксированных значений в `report/report.md` и ноутбуках проекта.

| Модель | Ключевые метрики | Примечание |
|--------|-------------------|------------|
| TabM (до обучения) | `ROC-AUC: 0.5570` | Baseline-точка до обучения |
| CatBoost (основной прогон) | `ROC-AUC: 0.8390` | `iterations=1000`, `depth=6`, `learning_rate=0.05` |
| CatBoost (порог 0.4) | `F1: 0.7813`, `Precision: 0.7043`, `Recall: 0.8774`, `Accuracy: 0.7375`, `PR-AUC: 0.8609` | Настройка порога классификации |
| TabM (обученная) | `ROC-AUC: 0.8418`, `F1: 0.7798`, `Accuracy: 0.7563` | Лучший `ROC-AUC` среди указанных |

## Линтеры

Проверка стиля и типичных ошибок в Python-коде (ноутбуки и `data/` исключены из flake8):

```bash
pip install -r requirements.txt

# Все линтеры (через Makefile)
make lint

# Или по отдельности
ruff check . --line-length 120
flake8 . --max-line-length=120 --extend-exclude=.venv,notebooks,data
```

В CI (`.github/workflows/ci.yml`) запускается `ruff check src/`.

## Отчёт

Финальный подробный отчёт: [`report/report.md`](report/report.md)
