"""
Часть 7 — Параллельная генерация и бюджет
==========================================
Из 3_persona_gen.py мы получили рабочий пайплайн: 50 персон через
make_client() + retry. Минус — он последовательный: 50 запросов × 2-3 с
на каждый = ~2 минуты.

Здесь делаем две вещи:
  1. Параллелим запросы через ThreadPoolExecutor — поскольку ботлнек
     это сеть (а не CPU), даже 10 воркеров дают ускорение ×8-10.
  2. Считаем токены и стоимость по полю response.usage — модель сама
     возвращает количество входных/выходных токенов в каждом ответе.

Запуск:
    python 7_batch_gen.py
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_client import get_model, make_client
from prompts import SYSTEM_PROMPT, USER_PROMPT
from schema import Persona

client = make_client()
MODEL = get_model()

# Тарифы в долларах за 1M токенов. По умолчанию — DeepSeek V4 Flash.
# Замени значения, если используешь другую модель.
PRICE_INPUT_PER_1M = 0.14
PRICE_OUTPUT_PER_1M = 0.28

N_PERSONAS = 30
MAX_WORKERS = 10


def generate_one() -> tuple[Persona, dict]:
    """Один запрос → (валидная Persona, словарь с usage)."""
    # TODO: дозаполни вызов client.chat.completions.create:
    #   - response_model=Persona
    #   - max_retries=3
    #   - temperature=0.9
    #   - messages из SYSTEM_PROMPT и USER_PROMPT
    # Дополнительно — попроси клиент вернуть raw-ответ через
    # `_raw_response=True` (если поддерживается обёрткой), либо считай
    # токены через отдельный вызов с `response_format={"type":"json_object"}`.
    #
    # Подсказка: обёртка make_client() умеет возвращать кортеж
    # (объект, completion), если передать with_completion=True.
    persona, completion = client.chat.completions.create(
        ...,
        with_completion=True,
    )
    usage = {
        "input_tokens": completion.usage.prompt_tokens,
        "output_tokens": completion.usage.completion_tokens,
    }
    return persona, usage


def estimate_cost(total_in: int, total_out: int) -> float:
    """Стоимость в долларах по двум счётчикам токенов."""
    return (total_in / 1_000_000) * PRICE_INPUT_PER_1M + (
        total_out / 1_000_000
    ) * PRICE_OUTPUT_PER_1M


def run_sequential(n: int) -> tuple[float, int, int]:
    """Последовательный baseline — для сравнения."""
    t0 = time.time()
    total_in, total_out = 0, 0
    for i in range(n):
        _, usage = generate_one()
        total_in += usage["input_tokens"]
        total_out += usage["output_tokens"]
        print(
            f"  [{i + 1:02d}/{n}] sequential — {usage['input_tokens']}→{usage['output_tokens']} токенов"
        )
    return time.time() - t0, total_in, total_out


def run_parallel(n: int, workers: int) -> tuple[float, int, int]:
    """Параллельная версия через ThreadPoolExecutor."""
    t0 = time.time()
    total_in, total_out = 0, 0
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(generate_one) for _ in range(n)]
        for fut in as_completed(futures):
            try:
                _, usage = fut.result()
                total_in += usage["input_tokens"]
                total_out += usage["output_tokens"]
                done += 1
                dt = time.time() - t0
                print(f"  [{done:02d}/{n}] parallel — за {dt:.1f}с накопили")
            except Exception as e:
                print(f"  ✗ {type(e).__name__}: {e}")
    return time.time() - t0, total_in, total_out


def main():
    print(f"Модель: {MODEL}, персон: {N_PERSONAS}\n")

    print("━━━ Параллельно ━━━")
    t_par, in_par, out_par = run_parallel(N_PERSONAS, MAX_WORKERS)

    print("\n━━━ Сводка ━━━")
    print(f"Параллельно ({MAX_WORKERS} worker'ов): {t_par:.1f}с")
    print(f"Входных токенов:  {in_par:>8d}")
    print(f"Выходных токенов: {out_par:>8d}")
    cost = estimate_cost(in_par, out_par)
    print(f"Стоимость:        ${cost:.4f}")
    print(f"На 1 персону:     ${cost / N_PERSONAS:.5f}")
    print()
    print(
        "На 1000 персон будет стоить примерно ${:.2f}".format(cost / N_PERSONAS * 1000)
    )


if __name__ == "__main__":
    main()
