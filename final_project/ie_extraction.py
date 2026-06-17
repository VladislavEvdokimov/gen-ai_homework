"""
IE — извлечение структурированных сущностей из транскрипта.
"""

import json
import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from schema import Transcript, Entity, Sentiment

load_dotenv()

CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

IE_PROMPT = """Ты — аналитик. Извлеки структурированные сущности из транскрипта фокус-группы на тему "Гибридные автомобили".

Для каждой реплики участника (не модератора) определи:
1. product_aspect — аспект продукта, который обсуждается (цена, надёжность, экология, дизайн, расход топлива, технология, комфорт, обслуживание и т.д.)
2. opinion — краткое изложение мнения участника об этом аспекте
3. sentiment — тональность (positive, neutral, negative)
4. quote — прямая цитата из реплики, подтверждающая мнение (до 200 символов)

Верни JSON-массив объектов с полями:
- persona_id
- persona_name
- product_aspect
- opinion
- sentiment (positive/neutral/negative)
- quote
"""


def extract_entities(transcript: Transcript) -> List[Entity]:
    """
    Извлекает сущности из транскрипта с помощью LLM.
    """
    # Форматируем транскрипт для подачи в LLM
    transcript_text = f"Тема: {transcript.topic}\n\n"
    for utt in transcript.utterances:
        speaker = "Модератор" if utt.persona_id == 0 else utt.persona_name
        transcript_text += f"[Раунд {utt.round}] {speaker}: {utt.text}\n\n"

    response = CLIENT.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": IE_PROMPT},
            {"role": "user", "content": f"Извлеки сущности из этого транскрипта:\n\n{transcript_text}"}
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)

    # Нормализация: может быть как массив, так и объект с ключом entities
    if isinstance(data, dict):
        for key in ["entities", "entity", "results"]:
            if key in data:
                data = data[key]
                break

    entities = []
    for item in data:
        try:
            entity = Entity(
                persona_id=item["persona_id"],
                persona_name=item["persona_name"],
                product_aspect=item["product_aspect"],
                opinion=item["opinion"],
                sentiment=Sentiment(item["sentiment"]),
                quote=item.get("quote", ""),
            )
            entities.append(entity)
        except (KeyError, ValueError) as e:
            print(f"  [WARN] Пропущена сущность: {e}")
            continue

    return entities


def save_entities(entities: List[Entity], path: str = "output/entities.json"):
    """Сохраняет сущности в JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([e.model_dump() for e in entities], f, ensure_ascii=False, indent=2)
    print(f"Сущности сохранены в {path}: {len(entities)} шт.")


def load_entities(path: str = "output/entities.json") -> List[Entity]:
    """Загружает сущности из JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Entity(**item) for item in data]


if __name__ == "__main__":
    from simulation import load_transcript
    transcript = load_transcript()
    entities = extract_entities(transcript)
    save_entities(entities)
    for e in entities:
        print(f"  [{e.sentiment.value}] {e.persona_name}: {e.product_aspect} — {e.opinion[:60]}...")