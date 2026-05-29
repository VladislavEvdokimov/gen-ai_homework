"""
Раунд 0.5 — Прачечная промптов
==============================
До того как мы потащим тяжёлую артиллерию (JSON mode + Pydantic), проверим
простой вопрос: насколько *промпт* влияет на то, парсится ли ответ как JSON?

Идея:
  1. Берём три варианта SYSTEM_PROMPT — от наивного до жёсткого.
  2. Прогоняем каждый по 5 запросов на «сыром» клиенте (без JSON mode).
  3. Считаем: какой процент ответов парсится JSON-ом, и в скольких возраст
     пришёл числом, а не строкой.

Запуск:
  python prompt_lab.py
"""

from __future__ import annotations

import json
import time

from llm_client import get_model, make_raw_client

client = make_raw_client()
MODEL = get_model()
USER_PROMPT = "Создай одну персону."

# Три варианта SYSTEM_PROMPT.
PROMPTS = {
    "naive": (
        "Ты генерируешь синтетические персоны покупателей российского "
        "e-commerce. Создай правдоподобного человека: укажи имя, возраст, "
        "город, месячный доход, род занятий, как часто он покупает онлайн "
        "и любимую категорию товаров. Ответ верни в формате JSON."
    ),
    "structured": (
        "Сгенерируй персону покупателя российского e-commerce. Верни JSON "
        "с полями: name (строка), age (число), city (строка), income_rub "
        "(число), occupation (строка), shopping_frequency (строка), "
        "preferred_category (строка)."
    ),
    "strict": (
        "Ты — генератор JSON. ВСЕГДА возвращай ТОЛЬКО валидный JSON-объект, "
        "без markdown, без комментариев, без текста до или после.\n"
        "Поля:\n"
        "- name (string)\n"
        "- age (integer, 18-75)\n"
        "- city (string)\n"
        "- income_rub (integer, 30000-500000)\n"
        "- occupation (string)\n"
        "- shopping_frequency (string)\n"
        "- preferred_category (string)\n"
        "ВСЕ числа — integer, не string. Без markdown-обёрток."
    ),
}

N_RUNS = 5


def try_parse(text: str):
    """Вернуть (parsed_ok, age_is_int, parsed_dict_or_None, raw_preview)."""
    raw = text.strip()
    # Минимальная зачистка markdown из-за того, что некоторые модели всё равно оборачивают код-блоком.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False, False, None, text[:80]
    age = data.get("age")
    return True, isinstance(age, int), data, text[:80]


def run_one(system_prompt: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": USER_PROMPT},
        ],
        temperature=0.9,
    )
    return resp.choices[0].message.content or ""


def main():
    print(f"Модель: {MODEL}")
    print(f"Запросов на каждый промпт: {N_RUNS}\n")

    results = {}
    for label, sys_prompt in PROMPTS.items():
        parsed, age_ok = 0, 0
        names, cities = set(), set()
        print(f"━━━ {label} ━━━")
        for i in range(N_RUNS):
            text = run_one(sys_prompt)
            ok, age_int, data, preview = try_parse(text)
            parsed += int(ok)
            age_ok += int(age_int)
            if data:
                if isinstance(data.get("name"), str):
                    names.add(data["name"])
                if isinstance(data.get("city"), str):
                    cities.add(data["city"])
            mark = "✓" if ok else "✗"
            age_mark = "int" if age_int else "—"
            print(
                f"  [{i + 1}/{N_RUNS}] {mark} parse={ok} age={age_mark} | {preview!r}"
            )
            time.sleep(0.3)
        results[label] = {
            "parsed": parsed,
            "age_int": age_ok,
            "uniq_names": len(names),
            "uniq_cities": len(cities),
        }
        print()

    print("━━━ Сводка ━━━")
    print(
        f"{'промпт':<12} {'JSON':>6} {'age=int':>9} {'уник.имён':>11} {'уник.городов':>14}"
    )
    for label, r in results.items():
        line = (
            f"{label:<12} {r['parsed']}/{N_RUNS:<3} "
            f"{r['age_int']}/{N_RUNS:<5}      {r['uniq_names']}/{N_RUNS}"
            f"           {r['uniq_cities']}/{N_RUNS}"
        )
        print(line)
    print()
    print("Что обсудить:")
    print("  - На «послушных» моделях (например, DeepSeek) разница в parse=ok")
    print("    исчезает уже на naive — модель умеет в JSON по умолчанию.")
    print("  - Но смотрите на «уник.имён» и «уник.городов»: если 1-2 на 5 —")
    print("    это mode collapse, и его промптом не лечится. Это про раунд 5/финал.")
    print("  - На моделях попроще (GPT-3.5, локальные 7B) различия в parse тоже")
    print("    есть — naive часто оборачивает markdown'ом, strict — нет.")
    print()
    print("Промпт снижает шум, но не убирает его. Переходим к JSON mode (раунд 1),")
    print("потом к Pydantic (раунд 2).")


if __name__ == "__main__":
    main()
