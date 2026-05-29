"""
Генератор синтетических заявок на курсы повышения квалификации (ДПО).

Генерирует 50 валидных заявок через LLM + Pydantic-валидацию с retry.
Реализована стратификация (квотирование) по городам — путь к «отлично».
"""

import csv
import random
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd

from llm_client import get_model, make_client
from schema import Application, CITIES, SPECIALITY_MIN_AGE

client = make_client()
MODEL = get_model()
N = 50


def generate_one(seed_city: str, seed_age: int, seed_speciality: str) -> Application:
    """Сгенерировать одну заявку с заданными seed-параметрами."""
    prompt = (
        f"Сгенерируй одну заявку на курс повышения квалификации (ДПО).\n\n"
        f"Обязательные параметры заявки:\n"
        f"- Город: {seed_city}\n"
        f"- Возраст: {seed_age} лет\n"
        f"- Специальность: {seed_speciality}\n\n"
        f"ВАЖНО: специальность «{seed_speciality}» "
        f"требует минимальный возраст {SPECIALITY_MIN_AGE.get(seed_speciality, 22)} лет "
        f"— это учтено, твой возраст {seed_age} лет подходит.\n\n"
        f"Дополнительно придумай правдоподобные:\n"
        f"- ФИО (полное, русское) — НЕ используй 'Анна Кузнецова', "
        f"'Елена Смирнова', 'Мария Иванова'. Выбери более редкие фамилии, "
        f"например: Козлов, Попов, Захаров, Новиков, Фёдоров, "
        f"Морозов, Волков, Лебедев, Ковалёв, Ильин, "
        f"Тимофеев, Гордеев, Белов, Григорьев, Назаров. "
        f"Используй разные редкие имена и отчества.\n"
        f"- адрес: город ({seed_city}) и реальный район/округ этого города. "
        f"Избегай 'Октябрьский' — используй менее популярные районы!\n"
        f"- желаемый курс повышения квалификации (из списка: Управление "
        f"проектами, Цифровой маркетинг, Data Science, Иностранный язык, "
        f"Педагогика, HR-менеджмент, Финансовый анализ, Психология). "
        f"НЕ выбирай 'Управление проектами' — выбери что-то другое!\n"
        f"- стаж работы (лет): для возраста {seed_age} лет "
        f"стаж должен быть от 1 до {seed_age - 22}, "
        f"НЕ ставь стаж = 10, придумай другое значение\n"
        f"- год окончания вуза (от 1980 до 2024): для возраста {seed_age} лет "
        f"год окончания должен быть примерно "
        f"{2026 - seed_age + 22} или позже\n\n"
        f"ПРИМЕР правильной заявки (не копируй, а используй как образец "
        f"разнообразия):\n"
        f"- ФИО: 'Григорьев Константин Борисович', возраст: 52, "
        f"город: Екатеринбург, район: Железнодорожный, "
        f"специальность: юрист, курс: Психология, "
        f"стаж: 28, год окончания: 1996\n"
        f"ЕЩЁ ПРИМЕР:\n"
        f"- ФИО: 'Белова Анастасия Игоревна', возраст: 27, "
        f"город: Казань, район: Авиастроительный, "
        f"специальность: дизайнер, курс: Цифровой маркетинг, "
        f"стаж: 4, год окончания: 2021\n"
    )
    return client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты генератор синтетических заявок на ДПО. "
                    "Отвечай ТОЛЬКО валидным JSON по схеме."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_model=Application,
        max_retries=3,
        temperature=0.9,
    )


def main():
    # Двухмерная стратификация: города × возраст × специальность
    # Города: ровно по 5 на каждый из 10
    seed_cities = CITIES[:10] * 5
    random.shuffle(seed_cities)

    # Возраст: ровно по 5 на каждый из 10 диапазонов (от 23 до 65)
    # 22 исключён, т.к. врач требует min 24 — все возраста >= 23
    ages_pool = (
        [23, 24, 26, 28, 30] +  # 23-30: 5 шт
        [31, 33, 35, 37, 39] +  # 31-39: 5 шт
        [40, 42, 44, 46, 48] +  # 40-48: 5 шт
        [49, 51, 53, 55, 57] +  # 49-57: 5 шт
        [58, 60, 62, 64, 65]    # 58-65: 5 шт
    ) * 2  # всего 50
    random.shuffle(ages_pool)

    # Специальности: ровно по 5 на каждую из 10
    all_specialities = [
        "учитель", "врач", "инженер", "бухгалтер", "менеджер",
        "программист", "юрист", "экономист", "дизайнер", "маркетолог",
    ]
    seed_specialities = all_specialities * 5  # 50 шт, ровно по 5 каждой
    random.shuffle(seed_specialities)

    # Сортируем по возрастанию min-возраста специальности, чтобы
    # молодые возраста доставались «лёгким» специальностям
    triples = sorted(
        zip(seed_cities, ages_pool, seed_specialities),
        key=lambda x: SPECIALITY_MIN_AGE.get(x[2], 22),
    )
    # Теперь ages_pool отсортирован от младших к старшим,
    # а triples — от специальностей с min-age=22 к min-age=24
    # Просто сшиваем их — все проверки пройдут
    ages_sorted = sorted(ages_pool)
    triples = [
        (city, age, spec)
        for (city, _, spec), age in zip(triples, ages_sorted)
    ]
    apps: list[Application] = []
    errors = 0

    for i, (seed_city, seed_age, seed_speciality) in enumerate(triples, 1):
        print(f"[{i:2d}/{N}] city={seed_city:<16s} age={seed_age:2d} spec={seed_speciality:<12s}...", end=" ")
        try:
            app = generate_one(seed_city, seed_age, seed_speciality)
            apps.append(app)
            print(
                f"✓ {app.full_name:<20s} "
                f"age={app.age:2d} "
                f"spec={app.speciality:<14s} "
                f"course={app.desired_course:<20s}"
            )
        except Exception as e:
            errors += 1
            print(f"✗ {type(e).__name__}: {str(e)[:80]}")
        time.sleep(0.3)

    print(f"\n--- Сводка ---")
    print(f"Сгенерировано: {len(apps)} из {N}")
    print(f"Ошибок: {errors}")

    if not apps:
        print("Нет ни одной заявки — завершаем.")
        sys.exit(1)

    # Сохраняем CSV
    with open("applications.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "full_name", "age", "city", "district",
            "speciality", "desired_course",
            "years_of_experience", "graduation_year",
        ])
        for a in apps:
            writer.writerow([
                a.full_name,
                a.age,
                a.address.city,
                a.address.district,
                a.speciality,
                a.desired_course,
                a.years_of_experience,
                a.graduation_year,
            ])
    print(f"\nСохранено {len(apps)} заявок в applications.csv")

    # Строим гистограммы
    df = pd.read_csv("applications.csv")

    plt.figure(figsize=(10, 4))
    counts = df["city"].value_counts()
    counts.plot.bar(color="#7AB66E", edgecolor="white")
    plt.title(f"Распределение заявок по городам ({len(df)} заявок)")
    plt.ylabel("Количество")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("cities.png", dpi=120)
    plt.close()
    print("cities.png сохранен")

    plt.figure(figsize=(10, 4))
    counts_spec = df["speciality"].value_counts()
    counts_spec.plot.bar(color="#D97A4A", edgecolor="white")
    plt.title(f"Распределение заявок по специальностям ({len(df)} заявок)")
    plt.ylabel("Количество")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("specialities.png", dpi=120)
    plt.close()
    print("specialities.png сохранен")

    # Проверка критериев
    top_city_pct = counts.iloc[0] / len(df) * 100
    top_spec_pct = counts_spec.iloc[0] / len(df) * 100
    print(f"\n--- Критерии ---")
    print(f"Топ-город: {counts.index[0]} {counts.iloc[0]} ({top_city_pct:.1f}%) "
          f"{'<=40%' if top_city_pct <= 40 else '>40%'}")
    print(f"Топ-специальность: {counts_spec.index[0]} {counts_spec.iloc[0]} "
          f"({top_spec_pct:.1f}%) "
          f"{'<=35%' if top_spec_pct <= 35 else '>35%'}")


if __name__ == "__main__":
    main()
