"""
Аспектный анализ: группировка извлечённых сущностей по аспектам.
"""

import json
import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
from schema import Entity, AspectGroup, AspectAnalysis

load_dotenv()

CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ASPECT_GROUPING_PROMPT = """Ты — аналитик маркетинговых исследований. У тебя есть список мнений участников фокус-группы о гибридных автомобилях.

Сгруппируй эти мнения по аспектам (темам). Например:
- цена / стоимость
- экология / окружающая среда
- надёжность / обслуживание
- расход топлива / экономичность
- дизайн / внешний вид
- технологии / инновации
- комфорт / удобство

Для каждой группы укажи:
- aspect — название аспекта (коротко, 1-2 слова)
- opinions — массив объектов того же формата (persona_id, persona_name, product_aspect, opinion, sentiment, quote), которые относятся к этому аспекту

Верни JSON с ключом "groups", содержащим массив групп.

Группируй минимум в 3, максимум в 8 групп. Каждая группа должна содержать хотя бы одну сущность.
"""


def analyze_aspects(entities: List[Entity]) -> AspectAnalysis:
    """
    Группирует извлечённые сущности по аспектам с помощью LLM.
    """
    # Сериализуем entities в текст
    entities_text = json.dumps([e.model_dump() for e in entities], ensure_ascii=False, indent=2)

    response = CLIENT.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": ASPECT_GROUPING_PROMPT},
            {"role": "user", "content": f"Сгруппируй эти мнения по аспектам:\n\n{entities_text}"}
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)

    # Извлекаем groups
    if isinstance(data, dict) and "groups" in data:
        groups_data = data["groups"]
    elif isinstance(data, list):
        groups_data = data
    else:
        # Попробуем найти groups в вложенных ключах
        groups_data = None
        for key in ["aspects", "categories", "clusters"]:
            if key in data:
                groups_data = data[key]
                break
        if groups_data is None:
            raise ValueError(f"Не удалось извлечь groups из ответа: {raw[:200]}...")

    groups = []
    for g in groups_data:
        opinions = []
        for o in g.get("opinions", g.get("entities", g.get("items", []))):
            from schema import Sentiment
            try:
                entity = Entity(
                    persona_id=o["persona_id"],
                    persona_name=o["persona_name"],
                    product_aspect=o.get("product_aspect", g.get("aspect", "")),
                    opinion=o["opinion"],
                    sentiment=Sentiment(o["sentiment"]),
                    quote=o.get("quote", ""),
                )
                opinions.append(entity)
            except (KeyError, ValueError) as e:
                print(f"  ⚠ Пропущена сущность в группе: {e}")
                continue

        groups.append(AspectGroup(
            aspect=g.get("aspect", g.get("category", g.get("name", ""))),
            opinions=opinions,
        ))

    return AspectAnalysis(groups=groups)


def save_aspect_analysis(analysis: AspectAnalysis, path: str = "output/aspects.json"):
    """Сохраняет анализ в JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(analysis.model_dump(), f, ensure_ascii=False, indent=2)
    print(f"Аспектный анализ сохранён в {path}: {len(analysis.groups)} групп")


def load_aspect_analysis(path: str = "output/aspects.json") -> AspectAnalysis:
    """Загружает анализ из JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return AspectAnalysis(**data)


if __name__ == "__main__":
    from ie_extraction import load_entities
    entities = load_entities()
    analysis = analyze_aspects(entities)
    save_aspect_analysis(analysis)
    for g in analysis.groups:
        print(f"  [{g.aspect}] {len(g.opinions)} мнений")