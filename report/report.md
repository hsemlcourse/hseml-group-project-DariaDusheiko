# Отчёт по проекту

**Студент:** Душейко Дарья Васильевна  
**Группа:** БИВ231

---

## 1. Введение и постановка задачи

- **Цель проекта:** предсказание вероятности возврата товара (`isReturned`) в задаче e-commerce, чтобы заранее выявлять заказы/товары/клиентов с высоким риском возврата.
- **Формулировка задачи:** бинарная классификация (`0` — не возврат, `1` — возврат).
- **Обоснование метрики качества:** основная метрика — **ROC-AUC**: она устойчива к дисбалансу классов (~55% возвратов в данных) и оценивает качество ранжирования вероятностей — это важно для формирования групп риска. Дополнительно считаются **F1**, **Precision**, **Recall**, **Accuracy** и **PR-AUC** (после выбора порога классификации).

---

## 2. Поиск и описание данных

- **Источник данных:** в репозитории присутствуют локальные `.p`-файлы в `data/raw` и `data/processed`; внешняя ссылка на исходный датасет - В качестве датасета используется набор транзакций маркетплейса с OSF https://osf.io/c793h/files/osfstorage.
- **Описание датасета (по файлам в репозитории):**
  - `data/raw/event_table_training.p`: `1,369,133 × 3` (`hash(variantID)`, `hash(customerId)`, `isReturned`)
  - `data/raw/event_table_testing.p`: `1,460,366 × 3`
  - `data/processed/customer_nodes_training.p`: `777,001 × 30`
  - `data/processed/customer_nodes_testing.p`: `825,598 × 30`
  - `data/processed/product_nodes_training.p`: `411,495 × 44`
  - `data/processed/product_nodes_testing.p`: `411,544 × 44`
- **Примеры признаков:** `yearOfBirth`, `isMale`, `salesPerCustomer`, `returnsPerCustomer`, `customerReturnRate`, `avgGbpPrice`, `avgDiscountValue`, `salesPerProduct`, `productReturnRate`, а также производные: `price_to_category_avg`, `discount_percent`, `customer_age`.
- **Баланс класса в event-таблицах (полный train):**
  - train: `1 → 757,227`, `0 → 611,906` (доля возвратов ~55,3%)
  - test: `1 → 795,366`, `0 → 665,000`
- **Подвыборка в `TABM.ipynb`:** для ускорения экспериментов использовано **20 000** транзакций из event-таблицы; после merge с узлами клиента/товара и очистки — **12 355** строк (доля возвратов ~55,1%).

---

## 3. Обработка и подготовка данных

- **Полная очистка** (`TABM.ipynb`, `сatboost_for_returns.ipynb`):
  - удаление полных дубликатов (`drop_duplicates`) — на подвыборке удалено **2** дубликата;
  - заполнение пропусков: медиана для числовых, мода для категориальных;
  - удаление признаков с нулевой дисперсией;
  - IQR-обрезка выбросов по `avgGbpPrice` и `avgDiscountValue`.
- **Feature engineering** (`TABM.ipynb`):
  - `price_to_category_avg` — отношение цены к средней по категории товара;
  - `discount_percent` — доля скидки от цены (%);
  - `customer_age` — возраст клиента (от `yearOfBirth`);
  - по важности (Random Forest): наибольший вклад у `discount_percent` (0,032), затем `price_to_category_avg` (0,024), `customer_age` (0,014).
- **Итоговая размерность признаков:** **70** числовых признаков в матрице `x_num` после merge и кодирования (подвыборка TABM).
- **Визуализации:** EDA в `TABM.ipynb` (баланс классов, `customerReturnRate` по классам, корреляции); итоговые графики качества CatBoost — в `report/` (см. раздел ниже).
- **Сплит данных:**
  - `сatboost_for_returns.ipynb`: `train_test_split(..., test_size=0.2, random_state=42, stratify=y)` → test **169 691** транзакция;
  - `TABM.ipynb`: stratified **64% / 16% / 20%** (train / val / test) → test **2 471** транзакция на подвыборке.
- **Data leakage:** признаки `customerReturnRate` и `productReturnRate` **пересчитываются только на обучающей выборке**, чтобы избежать утечки данных из будущего; `StandardScaler` обучается только на train.

---

## 4. Baseline-модель

- **Модель:** TabM со случайными весами до обучения (`TABM.ipynb`, подвыборка 20k).
- **Результат baseline:** `ROC-AUC: 0.4332` (ниже случайного уровня 0,5 на данном прогоне), `F1: 0.0000` — модель до обучения не ранжирует классы осмысленно.
- **Вывод:** необходимы полноценное обучение и/или классические модели (CatBoost, логистическая регрессия); baseline подтверждает, что без обучения нейросеть не даёт рабочего скоринга.

---

## 5. Эксперименты

Ниже — результаты из актуальных прогонов ноутбуков (после перезапуска).

### 5.1. CatBoost на полном hold-out (`сatboost_for_returns.ipynb`, test = 169 691)

| Модель | Гипотеза | Параметры / постановка | Результат (test) |
|--------|----------|-------------------------|------------------|
| CatBoost (основной) | Бустинг на полном наборе фичей | `iterations=1000`, `depth=6`, `learning_rate=0.05`, `eval_metric=AUC` | **ROC-AUC: 0.8390**, F1 @ 0.5: **0.7790**, Accuracy: **0.75** |
| CatBoost + порог **0.40** | Сдвиг порога улучшит F1 / recall | Подбор порога по validation | **F1: 0.7890**, Precision: **0.72**, Recall: **0.88**, Accuracy: **0.74** |
| CatBoost (сокращённый набор фичей) | Меньше фичей → ниже AUC | `iterations=500`, урезанные признаки | **ROC-AUC: 0.7016**, F1 @ 0.5: **0.7026**; при пороге **0.35**: F1 **0.7279** |

### 5.2. Оценка на event_test (`сatboost_test.ipynb`, test = 960 769)

| Метрика | Значение |
|---------|----------|
| ROC-AUC | **0.8373** |
| PR-AUC (Average Precision) | **0.8609** |
| F1 @ порог **0.40** | **0.7813** |
| Precision | **0.7043** |
| Recall | **0.8774** |
| Accuracy | **0.7375** |

### 5.3. Сравнение моделей на подвыборке (`TABM.ipynb`, test = 2 471)

| Модель | ROC-AUC | F1 | Accuracy | Precision | Recall |
|--------|---------|-----|----------|-----------|--------|
| Logistic Regression | 0.7092 | 0.7179 | 0.6378 | 0.6286 | 0.8369 |
| Random Forest | 0.6824 | 0.7127 | 0.5977 | 0.5874 | 0.9060 |
| XGBoost | 0.6685 | 0.6803 | 0.6333 | 0.6544 | 0.7083 |
| CatBoost | 0.6037 | 0.6935 | 0.5690 | 0.5700 | 0.8854 |
| Random Forest (tuned) | 0.7470 | 0.7513 | 0.6949 | 0.6816 | 0.8369 |
| **CatBoost (tuned)** | **0.7918** | **0.7543** | **0.7208** | **0.7319** | **0.7781** |
| TabM (best, epoch 8) | 0.6966 | 0.7103 | 0.5508 | 0.5508 | 1.0000 |

**Итог по экспериментам:** лучший **ROC-AUC на полных данных** — **CatBoost (0.8390)**. На подвыборке TABM лучший среди всех алгоритмов — **CatBoost (tuned), ROC-AUC 0.7918**; TabM на этом прогоне (**0.6966**) уступает бустингу. Подбор гиперпараметров выполнен через `RandomizedSearchCV` (Random Forest, CatBoost).

---

## 6. Финальная модель и интерпретируемость

- **Финальная модель:** **CatBoost** на полном hold-out (`сatboost_for_returns.ipynb`) — **ROC-AUC 0.8390**; для продакшен-порога рекомендован **0.40** (F1 **0.7890** на hold-out 169k, на event_test: F1 **0.7813**, PR-AUC **0.8609**).
- **Практический артефакт:** сохранённая модель `models/catboost_model.json` (`Logloss`, `eval_metric=AUC`).
- **Интерпретируемость:** в `сatboost_for_returns.ipynb` — `feature importance`; топ признаков связан с `customerReturnRate`, `productReturnRate`, ценой и скидкой. TabM на подвыборке показал слабее бустинг и склонность к предсказанию класса «возврат» (recall = 1.0 при низком precision).

---

## 7. Деплой

- **Интерфейс:** отдельный UI (Streamlit/бот) в репозитории не реализован.
- **API:** FastAPI/REST API отсутствует.
- **Запуск окружения:** `Dockerfile` поднимает Jupyter Notebook на порту `8888`.

Пример запуска:

```bash
docker build -t returns-project .
docker run --rm -p 8888:8888 returns-project
```

---

## 8. Заключение и выводы

- **Итоги:** задача бинарной классификации возвратов решена; на полных данных **CatBoost** даёт **ROC-AUC ≈ 0.84**, с порогом **0.40** — сбалансированные F1 / recall для бизнес-сценария «не пропустить возврат».
- **Сравнение с baseline:** TabM до обучения — **ROC-AUC 0.4332**; после обучения на подвыборке — **0.6966**, что ниже CatBoost на тех же данных (**0.7918** tuned).
- **Ограничения:** эксперименты TabM и сравнение 5+ моделей на подвыборке 20k; полный прогон TabM на 1.3M строк требует больше времени (CPU). Часть пайплайна живёт в ноутбуках, без вынесения в `src/`.
- **Возможные улучшения:** обучить TabM на полном train; вынести preprocessing и evaluation в `src/`; зафиксировать единый test-set и скрипт воспроизведения метрик.

---

## Графики из папки `report`

Графики построены в `notebooks/сatboost_test.ipynb` по предсказаниям CatBoost на тестовой выборке (`event_table_testing.p`). Дополнительно динамика метрик TabM по эпохам — в `notebooks/graphics.ipynb`.

### ROC curve
![ROC Curve](roc_curve.png)

**Описание:** зависимость True Positive Rate от False Positive Rate при переборе порогов; чем ближе кривая к левому верхнему углу, тем лучше ранжирование.

**Вывод:** ROC-AUC = **0.8373** — кривая существенно выше диагонали; модель хорошо отделяет возвраты от покупок при разных порогах и подходит для скоринга групп риска.

### Precision-Recall curve
![Precision Recall Curve](precision_recall.png)

**Описание:** trade-off между точностью (precision) и полнотой (recall) по классу «возврат»; площадь под кривой — PR-AUC (Average Precision).

**Вывод:** PR-AUC = **0.8609** — при фокусе на положительном классе модель сохраняет высокое качество несмотря на умеренный дисбаланс (~54% возвратов в test).

### Confusion matrix
![Confusion Matrix](confusion_matrix.png)

**Описание:** матрица ошибок при пороге **0.40**: строки — фактический класс, столбцы — предсказание (TN, FP, FN, TP).

**Вывод:** при пороге 0.4 — **257 848** верных «не возврат», **450 672** верных «возврат»; **189 248** ложных возвратов и **63 001** пропущенных возвратов. Recall по возвратам высокий (**0.8774**), что важно, если бизнес хочет минимизировать пропуски рискованных заказов.

### Calibration curve
![Calibration Curve](calibration_curve.png)

**Описание:** сравнение средней предсказанной вероятности с фактической долей возвратов по бинам; пунктир — идеальная калибровка.

**Вывод:** точки близки к диагонали — скоры можно интерпретировать как приближённую вероятность возврата; сильной перекалибровки на test не наблюдается.

### Predicted probability histogram
![Probability Histogram](proba_histogram.png)

**Описание:** распределение предсказанных вероятностей `P(isReturned=1)`; вертикальная линия — рабочий порог **0.40**.

**Вывод:** распределение бимодальное (масса около 0 и 1) — модель уверенно разделяет низкий и высокий риск; порог **0.40** (а не 0.50) даёт лучший F1 (**0.7813**) на event_test.

---

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
│   │   └── product_nodes_testing.p
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
