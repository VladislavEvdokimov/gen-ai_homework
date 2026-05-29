"""
Часть 9 — Потоковая выдача для интерактивных интерфейсов
==========================================================
Все предыдущие части мы ждали полного ответа — модель работает 2-3
секунды и присылает результат целиком. Для пакетного режима это нормально,
для интерфейса с пользователем — нет: ожидание на белом экране бесит.

Решение — потоковая выдача (streaming). Модель присылает ответ кусочками
(чаще всего по словам/токенам), и мы можем сразу выводить их на экран.
В чат-ботах вроде ChatGPT мы видим именно это — «печатающую машинку».

Здесь мы:
  1. Запускаем обычный запрос со stream=True.
  2. Печатаем токены по мере прихода.
  3. После полного ответа — собираем строку и валидируем Pydantic'ом
     (классическая «голова + хвост»: пользователь видит прогресс,
     валидация на финальном тексте).

Запуск:
    python 9_streaming.py
"""

from __future__ import annotations

import json
import sys
import time

from llm_client import get_model, make_raw_client
from prompts import SYSTEM_PROMPT, USER_PROMPT
from schema import Persona

client = make_raw_client()
MODEL = get_model()


def stream_and_validate() -> Persona:
    """Поток токенов на экран; в конце — валидация всего ответа."""
    print("⏵ ", end="", flush=True)
    t0 = time.time()

    # TODO: вызвать client.chat.completions.create с параметром stream=True.
    # Это вернёт итератор chunks вместо одного response.
    stream = ...

    chunks: list[str] = []
    for chunk in stream:
        # Каждый чанк имеет ту же структуру, что и обычный response,
        # но в choices[0].delta лежит только новый кусок.
        delta = chunk.choices[0].delta.content or ""
        chunks.append(delta)
        sys.stdout.write(delta)
        sys.stdout.flush()

    dt = time.time() - t0
    print(f"\n  (поток закончен за {dt:.1f}с, символов: {sum(len(c) for c in chunks)})")

    raw = "".join(chunks).strip()
    # Сервер мог обернуть ответ в маркдаун — чистим.
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    return Persona.model_validate(json.loads(raw))


def main():
    print(f"Модель: {MODEL}")
    print("Запрос идёт в потоковом режиме — следи за «печатающей машинкой».\n")
    try:
        p = stream_and_validate()
        print(f"\n✓ Валидная персона: {p.name}, {p.age}, {p.address.city}")
    except Exception as e:
        print(f"\n✗ {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
