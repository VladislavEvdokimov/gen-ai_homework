"""
Утилита «проверь мой JSON»
==========================
Запускается отдельно. Читает personas.json, проверяет каждую персону
против Pydantic-схемы и печатает человеко-читаемый отчёт.

Используется в конце семинара (раунд 5) и в домашнем задании —
чтобы студент мог сам себя проверить до сдачи.
"""

import json
import sys
from collections import Counter
from pydantic import ValidationError

try:
    from schema import Persona
except ImportError:
    print("✗ schema.py пустой или нет класса Persona. Начните с раунда 2.")
    sys.exit(1)


def main(path: str = "personas.json"):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"✗ Файл {path} не найден. Сначала сгенерируй.")
        return

    if not isinstance(data, list):
        print("✗ Ожидался список персон.")
        return

    valid, invalid = 0, 0
    problems = Counter()
    cities = Counter()
    occupations = Counter()
    names = Counter()

    for i, item in enumerate(data, 1):
        try:
            p = Persona(**item)
            valid += 1
            # p.city работает и для плоской схемы (раунды 2-4),
            # и для вложенной с address (раунд 4.5) — через @property.
            cities[p.city] += 1
            occupations[p.occupation] += 1
            names[p.name] += 1
        except ValidationError as e:
            invalid += 1
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"])
                problems[loc] += 1
                print(f"  #{i} ✗ {loc}: {err['msg']}")

    total = valid + invalid
    print(f"\n── Сводка ──")
    print(f"Всего:    {total}")
    print(f"Валидных: {valid}")
    print(f"С ошибками: {invalid}")

    if problems:
        print(f"\nТоп-проблемы:")
        for field, count in problems.most_common(5):
            print(f"  {count:3d}× {field}")

    if valid > 0:
        print(f"\n── Разнообразие ──")
        print(f"Уникальных имён: {len(names)} из {valid}")
        if names.most_common(1)[0][1] > 1:
            top = names.most_common(3)
            print(f"  ⚠ Повторы: {top}")
        print(f"Городов: {dict(cities)}")
        print(f"Профессий: {dict(occupations)}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "personas.json"
    main(path)
