# Семинар 2 — стартер

Тут лежит «скелет» — рабочее окружение и девять пронумерованных скриптов в правильном порядке.
Часть из них уже готовая (демонстрации), главный — `3_persona_gen.py` — *намеренно сломанный*.
Это работа на семинаре: дописать пустые места и увидеть, как меняется поведение модели.

## Установка (5 минут)

```bash
# 1. Виртуальное окружение
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Зависимости
pip install -r requirements.txt
# или:
# uv add --requirements requirements.txt

# 3. Скопируй .env.example в .env и впиши свой LLM_AUTH_TOKEN
cp .env.example .env
# затем открой .env в редакторе и впиши токен

# 4. Проверка, что endpoint работает
python -c "from llm_client import make_raw_client, get_model; \
           c = make_raw_client(); \
           r = c.chat.completions.create(model=get_model(), \
               messages=[{'role':'user','content':'ОК?'}], temperature=0); \
           print(r.choices[0].message.content)"
```

## Что делать, если застрял

- Перечитай комментарии `# TODO-раунд N` — там подсказка.
- Сравни свой `3_persona_gen.py` со `round0_expected_error.txt` — там расписаны типичные сбои.
- Спроси преподавателя.
- После пары в `../solution/` лежит эталонная версия — сравни.

## Заметка для финальных раундов (6-8)

Скрипты `7_batch_gen.py`, `8_tool_calling.py`, `9_streaming.py` импортируют
заполненную `Persona` из `schema.py` и зависят от `prompts.py`. Запускать
после того, как пройдены раунды 2-4.5.
