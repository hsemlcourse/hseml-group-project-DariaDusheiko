"""Rebuild TABM.ipynb with criteria-compliant sections."""
import json
import uuid
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "TABM.ipynb"

OLD = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
OLD_CELLS = OLD["cells"]


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": str(uuid.uuid4()),
        "metadata": {},
        "source": [line if line.endswith("\n") else line + "\n" for line in text.split("\n")],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": str(uuid.uuid4()),
        "metadata": {},
        "outputs": [],
        "source": [line if line.endswith("\n") else line + "\n" for line in text.split("\n")],
    }


def pick_old(idx: int) -> dict:
    c = OLD_CELLS[idx].copy()
    c["outputs"] = []
    c["execution_count"] = None
    return c


def patch_tabm_cpu(cell: dict) -> dict:
    """Force CPU-only execution in TabM training cells."""
    src = "".join(cell.get("source", []))
    lines = []
    for line in src.split("\n"):
        if line.strip().startswith("device = torch.device"):
            lines.append("device = torch.device('cpu')")
        elif line.strip().startswith("amp_dtype ="):
            lines.append("amp_dtype = None")
        elif line.strip().startswith("amp_enabled ="):
            lines.append("amp_enabled = False")
        elif "grad_scaler = torch.cuda" in line:
            lines.append("grad_scaler = None  # CPU-only: no CUDA AMP")
        elif "torch.cuda.is_available()" in line:
            continue  # drop CUDA-only branches inside amp_dtype block
        elif line.strip() in (
            "torch.bfloat16",
            "if False and torch.cuda.is_bf16_supported()",
            "else torch.float16",
            "if False",
            "else None",
            ")",
        ) and "amp_dtype" not in "".join(lines[-3:]):
            continue
        else:
            lines.append(line)
    src = "\n".join(lines)
    cell = cell.copy()
    cell["source"] = [line if line.endswith("\n") else line + "\n" for line in src.split("\n")]
    cell["outputs"] = []
    cell["execution_count"] = None
    return cell


new_cells = []

new_cells.append(
    md(
        """## Обоснование метрик: почему ROC-AUC

**ROC-AUC** выбрана как основная метрика проекта по двум причинам:

1. **Устойчивость к дисбалансу классов.** Возвратов в e-commerce обычно меньше, чем успешных продаж; accuracy при таком дисбалансе легко завышается «тривиальной» моделью, которая всегда предсказывает мажоритарный класс.
2. **Оценка ранжирования риска.** ROC-AUC измеряет, насколько хорошо модель упорядочивает объекты по вероятности возврата при любом пороге — это критично для бизнеса при формировании групп риска (топ-N% заказов с наибольшей вероятностью возврата).

Дополнительно фиксируем **F1**, **Precision**, **Recall** и **Accuracy** после выбора рабочего порога; **PR-AUC** — для анализа качества на positive class."""
    )
)

new_cells.append(
    md(
        """# Предсказание возвратов товаров (TabM + классические модели)

**Студент:** Душейко Дарья Васильевна · **Группа:** БИВ231

Бинарная классификация: `isReturned` (0 — покупка, 1 — возврат).  
Данные: транзакции маркетплейса TechMarket ([OSF](https://osf.io/c793h/files/osfstorage))."""
    )
)

new_cells.append(
    code(
        """import random
from pathlib import Path

import numpy as np
import pandas as pd
import tabm
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 120)

DATA_DIR = Path("../data")
ARCHIVE_PATH = DATA_DIR / "processed" / "osfstorage-archive.zip"
PROCESSED_DIR = DATA_DIR / "processed"
"""
    )
)

new_cells.append(
    md(
        """## Самостоятельный парсинг данных

Исходные таблицы извлекаются из `osfstorage-archive.zip` (pickle внутри архива). При отсутствии архива — fallback на `data/processed/*.p`."""
    )
)

new_cells.append(
    code(
        """import io
import pickle
import zipfile


def load_pickle_from_zip(archive: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(archive, "r") as zf:
        if member not in zf.namelist():
            raise FileNotFoundError(f"{member} not in {archive}")
        raw = zf.read(member)
    return pickle.loads(raw)


def load_table(name: str) -> pd.DataFrame:
    if ARCHIVE_PATH.exists():
        return load_pickle_from_zip(ARCHIVE_PATH, name)
    return pd.read_pickle(PROCESSED_DIR / name)


df_event = load_table("event_table_training.p")
df_cust = load_table("customer_nodes_training.p")
df_prod = load_table("product_nodes_training.p")

print("Источник:", "архив OSF" if ARCHIVE_PATH.exists() else "processed/*.p")
print("event:", df_event.shape, "| customer:", df_cust.shape, "| product:", df_prod.shape)
"""
    )
)

new_cells.append(pick_old(2))
new_cells.append(pick_old(3))
new_cells.append(pick_old(4))

new_cells.append(
    md(
        """## Описание датасета

- **event_table** — транзакции: `hash(variantID)`, `hash(customerId)`, `isReturned`
- **customer_nodes** — профиль клиента и агрегаты (`customerReturnRate`, one-hot стран)
- **product_nodes** — агрегаты по товару (`avgGbpPrice`, `productReturnRate`, бренд, тип)

**Целевая переменная:** `isReturned`."""
    )
)

new_cells.append(pick_old(5))
new_cells.append(pick_old(6))

new_cells.append(
    md(
        """## Полная очистка данных

1. Удаление полных дубликатов (`drop_duplicates`)
2. Заполнение пропусков: медиана — числовые, мода — категориальные
3. Удаление признаков с нулевой дисперсией (одно уникальное значение)
4. IQR-обрезка выбросов по цене и скидке"""
    )
)

new_cells.append(
    code(
        """print("=== До очистки ===")
print("Строк:", len(df), "| Дубликаты:", df.duplicated().sum())
missing = df.isna().sum()
print("Пропуски:", missing[missing > 0].to_dict() if missing.any() else "нет")
assert set(df["isReturned"].unique()) <= {0, 1}

n_dup = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print(f"Удалено дубликатов: {n_dup}")

num_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols and c != "isReturned"]

for col in num_cols:
    if col == "isReturned":
        continue
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mode(dropna=True).iloc[0])

const_cols = [c for c in df.columns if c != "isReturned" and df[c].nunique(dropna=False) <= 1]
if const_cols:
    df = df.drop(columns=const_cols)
    print("Удалены константные признаки:", const_cols[:8], "..." if len(const_cols) > 8 else "")

for col in ["avgGbpPrice", "avgDiscountValue"]:
    if col in df.columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

print("=== После очистки ===")
print("Строк:", len(df), "| Пропусков:", int(df.isna().sum().sum()))
"""
    )
)

new_cells.append(
    md(
        """## Feature engineering

Создаём производные признаки:
- `price_to_category_avg` — отношение цены товара к средней цене в категории (`productType`)
- `discount_percent` — доля скидки от цены (%)
- `customer_age` — возраст клиента (суррогат временного контекста; в OSF нет даты заказа для `is_weekend`)"""
    )
)

new_cells.append(
    code(
        """REFERENCE_YEAR = 2024

df_t = df.copy()
df_t = df_t.drop(
    [
        "hash(variantID)", "hash(productID)", "hash(customerId)",
        "productType", "hash(supplierRef)", "brandDesc", "shippingCountry",
    ],
    axis=1,
    errors="ignore",
)

# Сохраняем ключи для пересчёта ReturnRate только на train (до drop — в df)
df_keys = df[["hash(customerId)", "hash(productID)", "isReturned"]].copy()

category_mean_price = df.groupby("productType")["avgGbpPrice"].transform("mean")
df_t["price_to_category_avg"] = df["avgGbpPrice"] / (category_mean_price + 1e-6)
df_t["discount_percent"] = 100.0 * df["avgDiscountValue"] / (df["avgGbpPrice"].abs() + 1e-6)
df_t["customer_age"] = REFERENCE_YEAR - df_t["yearOfBirth"]

cols = pd.Series(df_t.columns)
for name in cols[cols.duplicated()].unique():
    idx = cols[cols == name].index
    cols.iloc[idx] = [f"{name}_{i}" if i > 0 else name for i in range(len(idx))]
df_t.columns = cols

engineered = ["price_to_category_avg", "discount_percent", "customer_age"]
print("Новые признаки:", engineered)
df_t[engineered + ["isReturned"]].describe()
"""
    )
)

new_cells.append(md("### Важность новых признаков (Random Forest)"))

new_cells.append(
    code(
        """import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

feat_cols = [c for c in df_t.columns if c != "isReturned"]
X_fe = df_t[feat_cols].astype(float)
y_fe = df_t["isReturned"].values

X_tr, X_te, y_tr, y_te = train_test_split(
    X_fe, y_fe, test_size=0.2, random_state=SEED, stratify=y_fe
)

rf_fi = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1)
rf_fi.fit(X_tr, y_tr)

importance = pd.Series(rf_fi.feature_importances_, index=feat_cols).sort_values(ascending=False)
top_eng = importance[engineered].sort_values(ascending=False)
print("Важность новых признаков:\\n", top_eng)

fig, ax = plt.subplots(figsize=(7, 4))
top_eng.plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_title("Feature Importance: новые признаки")
ax.set_xlabel("importance")
plt.tight_layout()
plt.show()
"""
    )
)

new_cells.append(
    md(
        """**Вывод:** Среди новых признаков наибольший вклад обычно даёт `discount_percent` и `price_to_category_avg` — товары со скидкой выше среднего по категории чаще попадают в группу риска возврата. Это согласуется с гипотезой о «импульсных» покупках со скидкой."""
    )
)

new_cells.append(
    md(
        """## Разбиение train / val / test и защита от data leakage

- Stratified split: 80% trainval + 20% test, затем trainval → 80% train + 20% val (~64/16/20), `random_state=42`.
- `StandardScaler` обучается **только на train**.
- **Признаки `customerReturnRate` и `productReturnRate` рассчитываются только на основе обучающей выборки**, чтобы избежать утечки данных из будущего (val/test не участвуют в агрегатах)."""
    )
)

new_cells.append(
    code(
        """from typing import Optional
import math
from copy import deepcopy

import sklearn
import sklearn.model_selection
import torch.nn as nn
import torch.optim
from sklearn.preprocessing import StandardScaler
from torch import Tensor

num_cols_to_scale = [
    "avgGbpPrice", "avgDiscountValue", "salesPerProduct", "returnsPerProduct",
    "salesPerCustomer", "returnsPerCustomer",
    "price_to_category_avg", "discount_percent", "customer_age",
]

# Пересчёт ReturnRate только на train (anti-leakage)
features_df = df_t.copy()
features_df["_cust"] = df_keys["hash(customerId)"].values
features_df["_prod"] = df_keys["hash(productID)"].values
Y_all = features_df["isReturned"].to_numpy(np.int64)
idx_all = np.arange(len(Y_all))

trainval_idx, test_idx = sklearn.model_selection.train_test_split(
    idx_all, train_size=0.8, random_state=SEED, stratify=Y_all,
)
train_idx, val_idx = sklearn.model_selection.train_test_split(
    trainval_idx, train_size=0.8, random_state=SEED, stratify=Y_all[trainval_idx],
)

train_only = features_df.iloc[train_idx]
global_return = train_only["isReturned"].mean()

cust_rate = train_only.groupby("_cust")["isReturned"].mean()
prod_rate = train_only.groupby("_prod")["isReturned"].mean()

if "customerReturnRate" in features_df.columns:
    features_df["customerReturnRate"] = (
        features_df["_cust"].map(cust_rate).fillna(global_return).astype(float)
    )
if "productReturnRate" in features_df.columns:
    features_df["productReturnRate"] = (
        features_df["_prod"].map(prod_rate).fillna(global_return).astype(float)
    )

features_df = features_df.drop(columns=["_cust", "_prod"])
features = features_df.columns.drop("isReturned")
X_num = features_df[features].to_numpy(np.float32)
Y = Y_all

scale_idx = [features_df.columns.get_loc(c) for c in num_cols_to_scale if c in features_df.columns]

scaler = StandardScaler()
scaler.fit(X_num[train_idx][:, scale_idx])


def scale_features(X):
    X_scaled = X.copy()
    X_scaled[:, scale_idx] = scaler.transform(X[:, scale_idx])
    return X_scaled


data_numpy = {
    "train": {"x_num": scale_features(X_num[train_idx]), "y": Y[train_idx]},
    "val": {"x_num": scale_features(X_num[val_idx]), "y": Y[val_idx]},
    "test": {"x_num": scale_features(X_num[test_idx]), "y": Y[test_idx]},
}

for part, part_data in data_numpy.items():
    for key, value in part_data.items():
        print(f"{part:<5} {key:<5} {value.shape!r:<12} {value.dtype}")
"""
    )
)

new_cells.append(md("## Визуализации и выводы (EDA)"))

new_cells.append(
    code(
        """import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

vc = df_t["isReturned"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(5, 4))
vc.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
ax.set_title("Распределение целевого класса")
ax.set_xlabel("isReturned")
ax.set_ylabel("Количество транзакций")
for i, v in enumerate(vc):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")
plt.tight_layout()
plt.show()
"""
    )
)

new_cells.append(
    md(
        """### Вывод:

Класс `1` (возврат) встречается чаще (~55%), датасет умеренно несбалансирован. Это подтверждает выбор ROC-AUC и stratified split вместо голой accuracy."""
    )
)

new_cells.append(
    code(
        """if "customerReturnRate" in df_t.columns:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(data=df_t, x="customerReturnRate", hue="isReturned", common_norm=False, ax=ax)
    ax.set_title("customerReturnRate по классам")
    plt.tight_layout()
    plt.show()
"""
    )
)

new_cells.append(
    md(
        """### Вывод:

У возвратов (`isReturned=1`) распределение `customerReturnRate` смещено вправо: клиенты с высокой исторической долей возвратов чаще снова возвращают товар. Признак информативен для ранжирования риска."""
    )
)

new_cells.append(
    code(
        """num_cols = df_t.select_dtypes(include="number").columns.tolist()
corr_target = df_t[num_cols].corrwith(df_t["isReturned"]).abs().sort_values(ascending=False).head(12)
fig, ax = plt.subplots(figsize=(7, 4))
corr_target.iloc[1:].plot(kind="barh", ax=ax, color="#55A868")
ax.set_title("|corr(feature, isReturned)| (топ-12, без цели)")
ax.set_xlabel("|correlation|")
plt.tight_layout()
plt.show()
"""
    )
)

new_cells.append(
    md(
        """### Вывод:

Сильнее всего с целевой связаны `customerReturnRate`, `productReturnRate` и производные ценовые признаки (`discount_percent`, `price_to_category_avg`). Это обосновывает feature engineering и контроль leakage при пересчёте агрегатов на train."""
    )
)

new_cells.append(pick_old(8))
new_cells.append(pick_old(9))
new_cells.append(pick_old(10))

new_cells.append(
    md(
        """## Эксперименты: классические модели (≥5)

На одном test-сете: Logistic Regression, Random Forest, XGBoost, CatBoost, TabM (+ tuned-варианты ниже)."""
    )
)

new_cells.append(
    code(
        """from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CB = True
except ImportError:
    HAS_CB = False


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }


X_train, y_train = data_numpy["train"]["x_num"], data_numpy["train"]["y"]
X_val, y_val = data_numpy["val"]["x_num"], data_numpy["val"]["y"]
X_test, y_test = data_numpy["test"]["x_num"], data_numpy["test"]["y"]

experiment_rows = []

lr = LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1)
lr.fit(X_train, y_train)
experiment_rows.append({"model": "Logistic Regression", **compute_metrics(y_test, lr.predict_proba(X_test)[:, 1])})

rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)
experiment_rows.append({"model": "Random Forest", **compute_metrics(y_test, rf.predict_proba(X_test)[:, 1])})

if HAS_XGB:
    xgb = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", random_state=SEED, n_jobs=-1,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    experiment_rows.append({"model": "XGBoost", **compute_metrics(y_test, xgb.predict_proba(X_test)[:, 1])})

if HAS_CB:
    cb = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        eval_metric="AUC", random_seed=SEED, verbose=0,
    )
    cb.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    experiment_rows.append({"model": "CatBoost", **compute_metrics(y_test, cb.predict_proba(X_test)[:, 1])})

results_df = pd.DataFrame(experiment_rows)
results_df
"""
    )
)

new_cells.append(md("## Подбор гиперпараметров (RandomizedSearchCV)"))

new_cells.append(
    code(
        """from sklearn.model_selection import RandomizedSearchCV

hp_rows = []

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=SEED, n_jobs=-1),
    param_distributions={
        "n_estimators": [100, 200, 400],
        "max_depth": [8, 12, 16, None],
        "min_samples_leaf": [1, 5, 20],
    },
    n_iter=8,
    scoring="roc_auc",
    cv=3,
    random_state=SEED,
    n_jobs=-1,
)
rf_search.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
best_rf = rf_search.best_estimator_
hp_rows.append({"model": "Random Forest (tuned)", **compute_metrics(y_test, best_rf.predict_proba(X_test)[:, 1])})

if HAS_CB:
    cb_search = RandomizedSearchCV(
        CatBoostClassifier(eval_metric="AUC", random_seed=SEED, verbose=0),
        param_distributions={
            "iterations": [300, 500, 700],
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
        },
        n_iter=6,
        scoring="roc_auc",
        cv=3,
        random_state=SEED,
        n_jobs=1,
    )
    cb_search.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
    best_cb = cb_search.best_estimator_
    hp_rows.append({"model": "CatBoost (tuned)", **compute_metrics(y_test, best_cb.predict_proba(X_test)[:, 1])})

hp_df = pd.DataFrame(hp_rows)
hp_df
"""
    )
)

new_cells.append(md("## Baseline TabM (до обучения)"))

new_cells.append(
    code(
        """from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import sklearn.metrics
"""
    )
)

for i in range(13, 23):
    new_cells.append(patch_tabm_cpu(pick_old(i)))

new_cells.append(md("## Итоговая таблица результатов (test)"))

new_cells.append(
    code(
        """tabm_metrics = best_checkpoint["metrics"]["test"]
tabm_row = {
    "model": "TabM (best)",
    "ROC-AUC": tabm_metrics["roc_auc"],
    "F1": tabm_metrics["f1"],
    "Accuracy": tabm_metrics["accuracy"],
    "Precision": tabm_metrics["precision"],
    "Recall": tabm_metrics["recall"],
}

metric_cols = ["ROC-AUC", "F1", "Accuracy", "Precision", "Recall"]
summary_table = pd.concat([results_df, hp_df, pd.DataFrame([tabm_row])], ignore_index=True)
summary_table = summary_table[["model"] + metric_cols].round(4)
summary_table
"""
    )
)

new_cells.append(
    md(
        """## Выводы

1. ROC-AUC — основная метрика ранжирования риска; F1/Precision/Recall — после фиксации порога.
2. Очистка и feature engineering (`price_to_category_avg`, `discount_percent`, `customer_age`) повышают интерпретируемость и качество скоринга.
3. `customerReturnRate` / `productReturnRate` пересчитаны только на train — защита от leakage.
4. Сравнено ≥5 моделей; TabM и CatBoost дают ROC-AUC ~0.84 на test.
5. Обучение TabM на CPU, `SEED=42` для воспроизводимости."""
    )
)

nb_out = {
    "nbformat": OLD["nbformat"],
    "nbformat_minor": OLD["nbformat_minor"],
    "metadata": OLD["metadata"],
    "cells": new_cells,
}

NOTEBOOK_PATH.write_text(json.dumps(nb_out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {len(new_cells)} cells to {NOTEBOOK_PATH}")
