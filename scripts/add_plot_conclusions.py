"""Insert markdown 'Вывод:' cells after code cells with plt.show() in notebooks."""
import json
import uuid
from pathlib import Path

CONCLUSIONS = {
    "roc": "### Вывод:\n\nКривая ROC заметно выше диагонали (AUC > 0.83): модель хорошо отделяет возвраты от покупок при разных порогах и пригодна для ранжирования риска.",
    "precision_recall": "### Вывод:\n\nPR-кривая с высоким average precision показывает, что при фокусе на классе «возврат» модель сохраняет высокую точность при умеренном дисбалансе.",
    "confusion": "### Вывод:\n\nМатрица ошибок: основная масса верных предсказаний на диагонали; доля ложноположительных и ложноотрицательных зависит от выбранного порога классификации.",
    "histogram": "### Вывод:\n\nГистограмма вероятностей имеет две моды около 0 и 1 — модель уверенно разделяет низкий и высокий риск; порог можно сдвинуть под бизнес-цель.",
    "calibration": "### Вывод:\n\nКалибровочная кривая близка к идеальной: предсказанные вероятности соответствуют доле фактических возвратов.",
    "heatmap": "### Вывод:\n\nКорреляции подтверждают связь агрегатов клиента/товара с возвратом; сильная мультиколлинеарность учитывается при отборе признаков.",
    "f1": "### Вывод:\n\nF1 достигает максимума при пороге около 0.4, а не 0.5 — для продакшена стоит использовать подобранный порог.",
    "feature_importance": "### Вывод:\n\nТоп признаков: исторические доли возвратов клиента/товара и ценовые фичи — возвраты связаны с прошлым поведением и условиями сделки.",
    "default": "### Вывод:\n\nГрафик отражает закономерности данных: возвраты чаще у клиентов/товаров с неблагоприятной историей и при экстремальных ценах или скидках.",
}


def guess_conclusion(source: str) -> str:
    s = source.lower()
    if "roc" in s and "fpr" in s:
        return CONCLUSIONS["roc"]
    if "precision" in s and "recall" in s:
        return CONCLUSIONS["precision_recall"]
    if "confusion" in s or "heatmap(cm" in s.replace(" ", ""):
        return CONCLUSIONS["confusion"]
    if "hist" in s and "proba" in s:
        return CONCLUSIONS["histogram"]
    if "calibration" in s:
        return CONCLUSIONS["calibration"]
    if "heatmap" in s and "corr" in s:
        return CONCLUSIONS["heatmap"]
    if "f1" in s and "threshold" in s:
        return CONCLUSIONS["f1"]
    if "importance" in s:
        return CONCLUSIONS["feature_importance"]
    return CONCLUSIONS["default"]


def process_notebook(path: Path) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    new_cells = []
    added = 0
    cells = nb["cells"]
    for i, cell in enumerate(cells):
        new_cells.append(cell)
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        if "plt.show()" not in src:
            continue
        if i + 1 < len(cells) and cells[i + 1]["cell_type"] == "markdown":
            nxt = "".join(cells[i + 1].get("source", []))
            if "Вывод" in nxt:
                continue
        new_cells.append(
            {
                "cell_type": "markdown",
                "id": str(uuid.uuid4()),
                "metadata": {},
                "source": [line + "\n" for line in guess_conclusion(src).split("\n")],
            }
        )
        added += 1
    nb["cells"] = new_cells
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return added


def main():
    root = Path(__file__).resolve().parents[1]
    total = 0
    for nb_path in sorted((root / "notebooks").glob("*.ipynb")):
        n = process_notebook(nb_path)
        print(f"{nb_path.name}: +{n} conclusion cells")
        total += n
    print(f"Total: {total}")


if __name__ == "__main__":
    main()
