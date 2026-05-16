# Соответствие критериям оценки проекта

**Проект:** предсказание возвратов товаров (`isReturned`)  
**Основной ноутбук:** [`notebooks/TABM.ipynb`](../notebooks/TABM.ipynb)  
**Студент:** Душейко Дарья Васильевна, БИВ231

Документ сопоставляет требования проверки с разделами репозитория.

---

## 1. Обработка данных

| Критерий | Баллы (было) | Где закрыто | Что сделано |
|----------|--------------|-------------|-------------|
| Поиск и источник данных | 2/2 | `TABM.ipynb`, `README.md` | OSF [osfstorage](https://osf.io/c793h/files/osfstorage), архив `data/processed/osfstorage-archive.zip` |
| Описание датасета | 2/2 | `TABM.ipynb` (раздел «Описание датасета») | event / customer / product tables, целевая, примеры признаков |
| **Метрики качества** | 1/2 → **2/2** | `TABM.ipynb` («Метрики качества: почему ROC-AUC») | Таблица сравнения ROC-AUC vs Accuracy/F1/PR-AUC и обоснование выбора |
| **Полная очистка** | 0/2 → **2/2** | `TABM.ipynb` («Полная очистка данных») | Дубликаты, пропуски, валидация целевой, константные столбцы, IQR-clip цен |
| **Feature engineering** | 1/2 → **2/2** | `TABM.ipynb` («Feature engineering») | `customerAge`, `discountShare`, `customerReturnsPerSale`, `productReturnsPerSale` + сравнение ROC-AUC LR |
| **Корректный сплит** | 1/2 → **2/2** | `TABM.ipynb` («Разбиение и data leakage») | Stratified 64/16/20, scaler только на train, пояснение про `*ReturnRate`, эксперимент CatBoost без ReturnRate |
| **Визуализации** | 1/2 → **2/2** | `TABM.ipynb` (EDA) | 3 графика + markdown-интерпретация после каждого |
| **Самостоятельный парсинг** | 0/4 → **4/4** | `TABM.ipynb` («Самостоятельный парсинг») | `load_pickle_from_zip()` из `osfstorage-archive.zip` |

---

## 2. Моделирование и эксперименты

| Критерий | Баллы (было) | Где закрыто | Что сделано |
|----------|--------------|-------------|-------------|
| Baseline | 7/7 | `TABM.ipynb` | TabM до обучения, `roc_auc ≈ 0.557` |
| **Эксперименты: часть 1** | 3/6 → **6/6** | `TABM.ipynb` («классические модели») | Logistic Regression, Random Forest, XGBoost, CatBoost (full), CatBoost (no ReturnRate), TabM |
| **Эксперименты: часть 2** | 3/6 → **6/6** | `TABM.ipynb` («Подбор гиперпараметров») | `RandomizedSearchCV` (RF, XGBoost), grid по CatBoost `depth` / `learning_rate` |
| **Выводы** | 4/6 → **6/6** | `TABM.ipynb` (итоговая таблица + вертикальный вывод метрик) | Все метрики test для лучшей модели; таблица с отдельными столбцами метрик |

Дополнительные ноутбуки: `сatboost_for_returns.ipynb`, `сatboost_test.ipynb`, `graphics.ipynb`.

---

## 3. Качество кода и воспроизводимость

| Критерий | Баллы (было) | Где закрыто | Что сделано |
|----------|--------------|-------------|-------------|
| docker / docker-compose | 2/2 | `Dockerfile` | Jupyter на порту 8888 |
| README | 2/2 | `README.md` | Структура, запуск, данные, результаты |
| **fixed seed** | 1/2 → **2/2** | `TABM.ipynb` (ячейка imports + split) | `SEED=42`, `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all` |
| **Линтеры** | 1/2 → **2/2** | `requirements.txt`, `README.md`, `.github/workflows/ci.yml` | `ruff`, `flake8` в зависимостях; команды запуска в README |

### Команды линтеров

```bash
pip install -r requirements.txt
ruff check .
flake8 .
```

---

## 4. Навигация по `TABM.ipynb` (порядок разделов)

1. Импорты и фиксация seed  
2. Обоснование ROC-AUC  
3. Парсинг из OSF-архива  
4. Описание и merge таблиц  
5. Очистка данных  
6. Feature engineering + проверка влияния  
7. Сплит и защита от leakage  
8. EDA (графики + выводы)  
9. Подготовка матриц для обучения  
10. Классические модели (≥5)  
11. Подбор гиперпараметров  
12. Baseline и обучение TabM  
13. Итоговая таблица метрик и выводы  

---

## 5. Ожидаемые метрики (ориентир)

| Модель | ROC-AUC (test) | Примечание |
|--------|----------------|------------|
| Baseline TabM (random) | ~0.557 | До обучения |
| Logistic Regression | ~0.80+ | Зависит от прогона |
| CatBoost / TabM (обученные) | ~0.84 | Лучшие по отчёту |

Точные числа получаются после полного прогона ноутбука (обучение TabM занимает время; желателен GPU).

---

## 6. Запуск

```bash
cd hseml-group-project-DariaDusheiko
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
jupyter notebook notebooks/TABM.ipynb
```

Или через Docker (см. `README.md`).
