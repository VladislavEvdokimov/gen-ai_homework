"""
Генерация синтетических персон для фокус-группы.
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from schema import Persona

load_dotenv()

CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

PERSONA_PROMPT = """Ты — демограф. Сгенерируй 6 синтетических персон для фокус-группы на тему "Гибридные автомобили".

Персоны должны различаться по:
- возрасту (от 20 до 70 лет)
- полу (male/female/non-binary)
- профессии
- уровню дохода (low/middle/high)
- месту жительства (city/suburb/rural)
- опыту владения автомобилем
- отношению к гибридам (positive/neutral/negative/undecided)
- чертам характера (коротко)

Каждая персона должна быть реалистичной и иметь:
- id (1..6)
- name
- age (18-90)
- gender (male/female/non-binary)
- occupation
- income_level (low/middle/high)
- residence (city/suburb/rural)
- car_experience (строка с описанием опыта)
- attitude_to_hybrid (positive/neutral/negative/undecided)
- personality_traits (краткое описание)

Верни JSON-массив из 6 объектов. Каждый объект строго по схеме Persona.
"""


def generate_personas(seed: int = 42) -> list[Persona]:
    """
    Генерирует 6 персон с использованием LLM.
    """
    response = CLIENT.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PERSONA_PROMPT},
            {"role": "user", "content": f"Используй seed {seed} для разнообразия. Сгенерируй 6 разных персон."}
        ],
        response_format={"type": "json_object"},
        temperature=0.8 + (seed % 10) / 100.0,
        seed=seed,
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)
    if isinstance(data, dict) and "personas" in data:
        data = data["personas"]
    personas = [Persona(**item) for item in data]
    return personas


def save_personas(personas: list[Persona], path: str = "input/personas.json"):
    """Сохраняет персоны в JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in personas], f, ensure_ascii=False, indent=2)
    print(f"Сохранено {len(personas)} персон в {path}")


def load_personas(path: str = "input/personas.json") -> list[Persona]:
    """Загружает персоны из JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Persona(**item) for item in data]


if __name__ == "__main__":
    personas = generate_personas(seed=42)
    save_personas(personas)
    for p in personas:
        print(f"  {p.id}. {p.name}, {p.age} лет, {p.occupation}, доход {p.income_level}, отношение к гибридам: {p.attitude_to_hybrid}")