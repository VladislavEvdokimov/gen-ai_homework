"""
Семинар 2 — Лаборатория отладки
================================
Генератор синтетических персон для российского e-commerce.

Цель скрипта: вернуть 50 валидных персон и сохранить их в personas.json.

СТАТУС: скрипт УЖЕ падает. На каждом раунде семинара мы чиним по одной проблеме.
        Заметки «# TODO-раунд N» помечают места, куда придут изменения.
"""

import json
import time

from llm_client import get_model, make_raw_client
from prompts import SYSTEM_PROMPT, USER_PROMPT

client = make_raw_client()
MODEL = get_model()

N_PERSONAS = 5


def generate_one() -> dict:
    """Один запрос к LLM → одна персона."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        temperature=0.9,
    )
    text = resp.choices[0].message.content

    # Костыль №1: иногда модель оборачивает ответ в ```json ... ```
    #             Срезаем то, что очевидно не-JSON.
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    # Костыль №2: иногда JSON парсится, но чего-то не хватает.
    #             try/except хоть какое-то.
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[WARN] JSON не распарсился: {e}. Сырой ответ:")
        print(text[:200])
        raise

    # TODO-раунд 0.5: до того, как трогать код — посмотреть, насколько
    #                 *промпт* влияет на парсинг. Запусти `python 1_prompt_lab.py`
    #                 и сравни три варианта SYSTEM_PROMPT (naive / structured /
    #                 strict). На «послушных» моделях смотри на «уник.имён».
    # TODO-раунд 1.5: после JSON mode попробовать few-shot — добавить в
    #                 SYSTEM_PROMPT 2 примера из seed_examples.json.
    #                 Готовая демонстрация: `python 2_few_shot_demo.py`.
    # TODO-раунд 2: добавить валидацию типов через Pydantic (см. schema.py).
    # TODO-раунд 3: добавить проверку, что город существует (CITIES + field_validator).
    # TODO-раунд 4: добавить Literal на occupation и preferred_category.
    # TODO-раунд 4.5: расширить Persona вложенной моделью Address
    #                 (поля city + district). Pydantic валидирует рекурсивно —
    #                 модели часто ошибаются на вложенных структурах, и это
    #                 хороший разгон перед раундом 5 с retry.
    return data


def main():
    personas = []
    for i in range(N_PERSONAS):
        print(f"[{i + 1}/{N_PERSONAS}] запрос...")
        try:
            p = generate_one()
            personas.append(p)
            # Печатаем ТИП каждого поля — чтобы было видно, где плывёт.
            print(
                f"  → name={p.get('name')!r} "
                f"age={p.get('age')!r} (type={type(p.get('age')).__name__})"
            )
        except Exception as e:
            print(f"  ✗ упало: {type(e).__name__}: {e}")
        time.sleep(0.3)  # не долбим API

    # TODO-раунд 5: перейти на make_client() из llm_client.py с параметром
    #               response_model=Persona и max_retries=3, чтобы каждая
    #               невалидная персона догонялась до валидной автоматически.
    #               (Это та же идея, что в пакете `instructor`, но через
    #               нашу обёртку — instructor ставить НЕ нужно.)
    # TODO-раунд 5.5: запусти `python 4_stress_test.py` и убедись, что retry
    #                 НЕ магия — на конфликте «промпт vs схема» max_retries=5
    #                 потратит токены и всё равно вернёт ошибку.
    # TODO-финал: после генерации полных 50 — `python 5_analysis.py` даёт
    #             расширенный отчёт (кросс-таблица, дубликаты, корреляции).

    print(f"\nСгенерировано: {len(personas)} из {N_PERSONAS}")
    with open("personas.json", "w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)
    print("Сохранено в personas.json")


if __name__ == "__main__":
    main()
