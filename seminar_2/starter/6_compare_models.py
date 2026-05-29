"""
Если у вас есть доступ к двум эндпоинтамам (например, self-hosted Qwen и
публичный gpt-4o-mini), полезно прогнать одну и ту же связку «промпт +
схема + retry» через обе и сравнить:
  - доля валидных персон,
  - mode collapse (есть ли «город-чемпион», который выгребает половину),
  - среднее время одного запроса.

Как настроить:
  Заполните .env_compare двумя блоками переменных (см. .env_compare.example).
  Скрипт переключается между ними и прогоняет одну и ту же выборку.

Запуск:
  python compare_models.py
"""

from __future__ import annotations

import os
import time
from collections import Counter

from llm_client import JsonClient, _make_openai_client
from prompts import SYSTEM_PROMPT, USER_PROMPT
from schema import Persona

N_PER_MODEL = 10  # на разминке — 10; для надёжных цифр поднимите до 50


def make_client_for(env: dict) -> tuple[JsonClient, str]:
    """Создать клиента, подменив переменные окружения на лету."""
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update({k: v for k, v in env.items() if v is not None})
    try:
        c = JsonClient(_make_openai_client())
        model = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
        return c, model
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_batch(label: str, env: dict) -> dict:
    print(f"\n━━━ {label} ━━━")
    client, model = make_client_for(env)
    print(f"  model={model}, base={env.get('LLM_BASE_URL') or 'public OpenAI'}")
    valid = 0
    cities, occupations = Counter(), Counter()
    times = []
    for i in range(N_PER_MODEL):
        t0 = time.time()
        try:
            p = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT},
                ],
                response_model=Persona,
                max_retries=3,
                temperature=0.9,
            )
            dt = time.time() - t0
            times.append(dt)
            valid += 1
            cities[p.city] += 1
            occupations[p.occupation] += 1
            print(f"  [{i + 1:02d}] ✓ {dt:.1f}c {p.city}/{p.occupation}")
        except Exception as e:
            dt = time.time() - t0
            times.append(dt)
            print(f"  [{i + 1:02d}] ✗ {dt:.1f}c {type(e).__name__}")

    top_city = cities.most_common(1)[0] if cities else (None, 0)
    top_occ = occupations.most_common(1)[0] if occupations else (None, 0)
    avg = sum(times) / len(times) if times else 0
    return {
        "label": label,
        "valid": valid,
        "n": N_PER_MODEL,
        "avg_time": avg,
        "top_city": top_city,
        "top_occ": top_occ,
        "city_diversity": len(cities),
        "occ_diversity": len(occupations),
    }


def main():
    # Конфигурации двух моделей. ЗАПОЛНИТЕ под свой стенд.
    config_a = {
        "LLM_BASE_URL": os.environ.get("MODEL_A_BASE_URL", ""),
        "LLM_AUTH_TOKEN": os.environ.get("MODEL_A_TOKEN", ""),
        "LLM_MODEL": os.environ.get("MODEL_A_NAME", "llm"),
    }
    config_b = {
        "LLM_BASE_URL": os.environ.get("MODEL_B_BASE_URL", ""),
        "LLM_AUTH_TOKEN": os.environ.get("MODEL_B_TOKEN", ""),
        "LLM_MODEL": os.environ.get("MODEL_B_NAME", "gpt-4.1-mini"),
    }

    if not config_a["LLM_BASE_URL"] and not config_b["LLM_BASE_URL"]:
        print("ℹ Не заданы MODEL_A_BASE_URL / MODEL_B_BASE_URL.")
        print("  Задайте их через окружение или .env_compare и перезапустите.")
        print("  Пример (self-hosted vs OpenAI):")
        print("    export MODEL_A_BASE_URL=https://inference.parsers360.ru:10443/v1")
        print("    export MODEL_A_TOKEN=...")
        print("    export MODEL_A_NAME=llm")
        print("    export MODEL_B_NAME=gpt-4.1-mini")
        print("    # для B base_url оставьте пустым → пойдёт в публичный OpenAI")
        return

    a = run_batch("Model A", config_a)
    b = run_batch("Model B", config_b)

    print("\n━━━ Итог ━━━")
    valid_a = f"{a['valid']}/{a['n']}"
    valid_b = f"{b['valid']}/{b['n']}"
    city_a = f"{a['top_city'][0]}({a['top_city'][1]})"
    city_b = f"{b['top_city'][0]}({b['top_city'][1]})"
    occ_a = f"{a['top_occ'][0]}({a['top_occ'][1]})"
    occ_b = f"{b['top_occ'][0]}({b['top_occ'][1]})"
    print(f"{'Метрика':<28} {'A':>14} {'B':>14}")
    print(f"{'Валидных':<28} {valid_a:>14} {valid_b:>14}")
    print(f"{'Среднее время, c':<28} {a['avg_time']:>14.1f} {b['avg_time']:>14.1f}")
    print(f"{'Уник. городов':<28} {a['city_diversity']:>14} {b['city_diversity']:>14}")
    print(f"{'Уник. профессий':<28} {a['occ_diversity']:>14} {b['occ_diversity']:>14}")
    print(f"{'Топ-город':<28} {city_a:>14} {city_b:>14}")
    print(f"{'Топ-профессия':<28} {occ_a:>14} {occ_b:>14}")
    print("\nОбсудить с группой:")
    print("  - какая модель надёжнее на валидации, какая быстрее?")
    print("  - у какой сильнее mode collapse — у «большой» или у «маленькой»?")
    print("  - стоит ли разница в цене разницы в качестве для этой задачи?")


if __name__ == "__main__":
    main()
