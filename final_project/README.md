# Финальный проект: Симуляция фокус-группы (Трек A — homo silicus)

## Описание

Симуляция фокус-группы на тему **«Гибридные автомобили»** с использованием LLM-агентов.

**Пайплайн:**

1. **Синтетические персоны** — генерация 6 демографических профилей
2. **Мультиагентная симуляция** — модератор + 6 персон обсуждают гибридные авто (4 раунда)
3. **IE — извлечение сущностей** — структурированные мнения из транскрипта
4. **Аспектный анализ** — группировка мнений по темам (цена, экология, надёжность...)
5. **Map-Reduce** — суммаризация по каждому аспекту
6. **LLM-as-Judge** — оценка качества и проверка галлюцинаций

## Техники курса (6 шт.)

1. Синтетические персоны
2. Мультиагент
3. IE — извлечение структурированных сущностей
4. Аспектный анализ
5. Map-Reduce
6. LLM-as-Judge

## Установка

```bash
pip install -r requirements.txt
```

## Настройка

1. Скопируйте `.env.example` в `.env`:
```bash
cp .env.example .env
```

2. Вставьте ваш OpenAI API-ключ в `.env`:
```
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
OPENAI_JUDGE_MODEL=deepseek-chat
```

## Запуск

### Полный пайплайн (один прогон):
```bash
python pipeline.py
```

### С параметрами:
```bash
python pipeline.py 42 4   # seed=42, 4 раунда
```

### Eval (15 тестов — 3 темы × 5 seed):
```bash
python eval.py
```

### Принудительный перезапуск eval:
```bash
python eval.py --force
```

### Отдельные шаги:
```bash
python personas.py        # генерация персон
python simulation.py      # симуляция дискуссии
python ie_extraction.py   # извлечение сущностей
python aspect_analysis.py # аспектный анализ
python map_reduce.py      # map-reduce суммаризация
python judge.py           # LLM-as-Judge
```

## Структура проекта

```
├── README.md
├── requirements.txt
├── .env.example
├── schema.py              # Pydantic-схемы с field_validator
├── personas.py             # Генерация персон
├── simulation.py           # Мультиагентная симуляция
├── ie_extraction.py        # IE — извлечение сущностей
├── aspect_analysis.py      # Аспектный анализ
├── map_reduce.py           # Map-Reduce суммаризация
├── judge.py                # LLM-as-Judge
├── pipeline.py             # Главный пайплайн
├── eval.py                 # Eval на 15 тестах
├── input/
│   └── personas.json       # Сгенерированные персоны
├── output/
│   ├── transcript.json     # Транскрипт дискуссии
│   ├── entities.json       # Извлечённые сущности
│   ├── aspects.json        # Аспектный анализ
│   ├── summary.json        # Map-Reduce отчёт
│   ├── judge_report.json   # Оценка Judge
│   ├── eval_results.csv    # Результаты eval
│   ├── eval_cases.json     # Детальные результаты eval
│   └── trace.json          # Трейс выполнения
└── отчёт.md
```

## Валидация (field_validator)

- `Persona.age`: возраст 18–90
- `Persona.name`: непустое имя
- `JudgeEvaluation.*_score`: оценка 1–5
- `Utterance.round`: номер раунда 1–20
- `Entity.sentiment`: только positive/neutral/negative