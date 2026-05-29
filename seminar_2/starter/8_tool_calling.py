"""
Часть 8 — Вызов инструментов как альтернатива JSON-mode
========================================================
До сих пор мы получали структурированный вывод через JSON-mode:
сервер обещает, что ответ будет валидным JSON, а соответствие схеме
гарантирует наш Pydantic + retry.

Есть альтернатива — механизм tool calling (вызов инструментов). Идея:
вместо «верни JSON» мы говорим модели «вот функция register_persona(...)
с такими-то параметрами; вызови её». Модель отвечает специальным полем
tool_calls со словарём аргументов — это и есть наша структура.

Плюсы:
  • Параметры описываются как обычная JSON Schema — без отдельной
    инструкции в промпте.
  • На многих сервисах (OpenAI, Anthropic) это нативный путь и работает
    надёжнее, чем JSON-mode.
  • Удобно для нескольких инструментов: одна модель — несколько схем,
    и она выбирает, какую вызвать.

Минусы:
  • Поддержка зависит от сервера. Self-hosted Qwen/vLLM могут не уметь.
  • Сложнее парсить — нужно лезть в choices[0].message.tool_calls.

Запуск:
    python 8_tool_calling.py
"""
from __future__ import annotations

import json

from llm_client import make_raw_client, get_model
from schema import Persona

client = make_raw_client()
MODEL = get_model()


# JSON-Schema описания инструмента. Берём прямо из Pydantic-модели —
# Persona.model_json_schema() возвращает совместимый JSON Schema.
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "register_persona",
        "description": "Зарегистрировать сгенерированную персону покупателя.",
        # TODO: подставить сюда Persona.model_json_schema()
        "parameters": ...,
    },
}


def generate_via_tools() -> Persona:
    """Запрос с принудительным вызовом register_persona()."""
    # TODO: вызвать client.chat.completions.create с параметрами:
    #   - tools=[TOOL_SCHEMA]
    #   - tool_choice={"type": "function", "function": {"name": "register_persona"}}
    #   - messages: system из prompts.SYSTEM_PROMPT + user "Создай одну персону."
    response = ...

    # Извлекаем аргументы вызова и валидируем Pydantic'ом.
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    return Persona.model_validate(args)


def main():
    print(f"Модель: {MODEL}")
    print("Сравниваем три способа получить структурированный вывод:")
    print("  1. JSON-mode (то, что делали раньше)")
    print("  2. Tool calling (этот файл)")
    print("Запускаем по 3 раза.\n")

    print("━━━ Tool calling ━━━")
    for i in range(3):
        try:
            p = generate_via_tools()
            print(f"  [{i+1}/3] ✓ {p.name}, {p.age}, {p.address.city}")
        except Exception as e:
            print(f"  [{i+1}/3] ✗ {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
