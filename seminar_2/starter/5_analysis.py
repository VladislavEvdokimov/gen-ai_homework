"""
Финал семинара — расширенный анализ качества данных
====================================================
После раунда 5 у нас есть `personas.json` с 50 валидными персонами.
Раньше мы смотрели одну гистограмму возраста — и заканчивали. Здесь
копаем глубже: что именно пошло не так в распределении?

Что считаем:
  1. Гистограмма возрастов — была и раньше.
  2. Распределение по городам и профессиям (бары) — найдём mode collapse.
  3. Топ-N повторяющихся имён — другая грань collapse (модель любит «Анну»).
  4. Кросс-таблица город × профессия — есть ли нереалистичные комбинации?
  5. Boxplot доход × профессия — модель умеет связывать поля или просто
     генерит независимо?

На выходе:
  - ages.png         — гистограмма возрастов
  - cities.png       — распределение по городам
  - occupations.png  — распределение по профессиям
  - income_by_occupation.png — boxplot
  - report.md        — текстовая сводка для обсуждения

Запуск:
  python analysis.py [personas.json]
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        sys.exit("Файл пустой — сначала прогоните persona_gen_solution.py")
    # Поддерживаем и плоскую (раунды 2-4), и вложенную (раунд 4.5) Persona.
    # Если внутри есть address: {city, district} — распаковываем наверх.
    flat = []
    for item in data:
        row = dict(item)
        if isinstance(row.get("address"), dict):
            addr = row.pop("address")
            row.setdefault("city", addr.get("city"))
            row.setdefault("district", addr.get("district"))
        flat.append(row)
    return pd.DataFrame(flat)


def plot_hist_ages(df: pd.DataFrame, out: str):
    plt.figure(figsize=(8, 4))
    plt.hist(df["age"], bins=12, color="#4A90D9", edgecolor="white")
    plt.xlabel("Возраст")
    plt.ylabel("Число персон")
    plt.title(f"Распределение возраста ({len(df)} персон)")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_bar(series: pd.Series, title: str, out: str, color="#4A90D9"):
    counts = series.value_counts()
    plt.figure(figsize=(9, 4))
    counts.plot.bar(color=color, edgecolor="white")
    plt.title(title)
    plt.ylabel("Число персон")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return counts


def plot_income_by_occupation(df: pd.DataFrame, out: str):
    if "income_rub" not in df.columns or "occupation" not in df.columns:
        return
    groups = df.groupby("occupation")["income_rub"].apply(list)
    plt.figure(figsize=(10, 4))
    # labels= переименован в tick_labels в matplotlib 3.9; используем общий
    # подход — задаём положения, потом xticks с подписями.
    positions = range(1, len(groups) + 1)
    plt.boxplot(list(groups.values), positions=list(positions), vert=True)
    plt.xticks(list(positions), list(groups.index), rotation=30, ha="right")
    plt.ylabel("Доход, ₽/мес")
    plt.title("Доход × профессия")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def cross_table(df: pd.DataFrame) -> pd.DataFrame:
    if "city" not in df.columns or "occupation" not in df.columns:
        return pd.DataFrame()
    return pd.crosstab(df["city"], df["occupation"])


def write_report(df: pd.DataFrame, out: str):
    n = len(df)
    lines = [f"# Отчёт по {n} персонам\n"]

    # Топ городов
    cities = df["city"].value_counts()
    top_city_pct = cities.iloc[0] / n * 100
    lines.append("## Города\n")
    lines.append(f"- Уникальных: {len(cities)} из 8 разрешённых")
    lines.append(f"- Топ-1: **{cities.index[0]}** — {cities.iloc[0]} ({top_city_pct:.0f}%)")
    if top_city_pct > 40:
        lines.append(f"- ⚠ Превышен порог 40% → mode collapse по городам")
    lines.append("")

    # Топ профессий
    occ = df["occupation"].value_counts()
    top_occ_pct = occ.iloc[0] / n * 100
    lines.append("## Профессии\n")
    lines.append(f"- Уникальных: {len(occ)} из 8 разрешённых")
    lines.append(f"- Топ-1: **{occ.index[0]}** — {occ.iloc[0]} ({top_occ_pct:.0f}%)")
    if top_occ_pct > 35:
        lines.append(f"- ⚠ Превышен порог 35% → mode collapse по профессиям")
    lines.append("")

    # Дубликаты имён
    names = df["name"].value_counts()
    dupes = names[names > 1]
    lines.append("## Имена\n")
    lines.append(f"- Уникальных: {len(names)} из {n} ({len(names)/n*100:.0f}%)")
    if len(dupes):
        lines.append(f"- Повторы: {dict(dupes.head(5))}")
    else:
        lines.append("- Повторов нет")
    lines.append("")

    # Кросс-таблица
    ct = cross_table(df)
    if not ct.empty:
        lines.append("## Кросс-таблица город × профессия\n")
        lines.append("```")
        lines.append(ct.to_string())
        lines.append("```")
        # Подозрительные комбо — пустые ячейки в крупных городах
        for city in cities.head(2).index:
            row = ct.loc[city] if city in ct.index else None
            if row is not None:
                empty = row[row == 0].index.tolist()
                if empty:
                    lines.append(f"- В **{city}** ни одного: {', '.join(empty)}")
        lines.append("")

    # Доход × профессия
    if "income_rub" in df.columns:
        med = df.groupby("occupation")["income_rub"].median().sort_values(ascending=False)
        lines.append("## Медианный доход по профессиям\n")
        for occ_name, m in med.items():
            lines.append(f"- {occ_name}: {int(m):,} ₽".replace(",", " "))
        # Sanity-check: студент с доходом > 200k или пенсионер > 100k — звоночек
        if "студент" in df["occupation"].values:
            stud_max = df[df["occupation"] == "студент"]["income_rub"].max()
            if stud_max > 200_000:
                lines.append(f"- ⚠ Студент с доходом {stud_max:,} ₽ — модель не связала поля".replace(",", " "))
        lines.append("")

    Path(out).write_text("\n".join(lines), encoding="utf-8")


def main(path: str = "personas.json"):
    df = load(path)
    print(f"Загружено: {len(df)} персон из {path}")

    plot_hist_ages(df, "ages.png")
    c = plot_bar(df["city"], "Распределение по городам", "cities.png", "#7AB66E")
    o = plot_bar(df["occupation"], "Распределение по профессиям", "occupations.png", "#D97A4A")
    plot_income_by_occupation(df, "income_by_occupation.png")
    write_report(df, "report.md")

    print("\nСохранено:")
    for f in ("ages.png", "cities.png", "occupations.png",
              "income_by_occupation.png", "report.md"):
        if Path(f).exists():
            print(f"  - {f}")

    print(f"\nТоп-город: {c.index[0]} ({c.iloc[0]}/{len(df)})")
    print(f"Топ-профессия: {o.index[0]} ({o.iloc[0]}/{len(df)})")
    print("\nДальше — открыть report.md и обсудить с группой:")
    print("  - где collapse, какое поле «слиплось» сильнее всего?")
    print("  - есть ли нереалистичные комбинации в кросс-таблице?")
    print("  - модель связывает доход с профессией или генерит независимо?")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "personas.json"
    main(path)
