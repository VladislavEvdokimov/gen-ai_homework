"""
Мультиагентная симуляция фокус-группы.
"""

import json
import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from schema import Persona, Transcript, Utterance

load_dotenv()

CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def build_system_prompt(persona: Persona) -> str:
    """Строит system prompt для персоны."""
    return f"""Ты — участник фокус-группы. Твоя роль:

Имя: {persona.name}
Возраст: {persona.age}
Пол: {persona.gender}
Профессия: {persona.occupation}
Уровень дохода: {persona.income_level}
Место жительства: {persona.residence}
Опыт с авто: {persona.car_experience}
Отношение к гибридам: {persona.attitude_to_hybrid}
Черты характера: {persona.personality_traits}

Отвечай на реплики модератора и других участников, оставаясь в своей роли.
Твои ответы должны отражать твой возраст, профессию, доход и жизненный опыт.
Будь естественным/естественной, как в реальной фокус-группе. Пиши 2-4 предложения за раз.
Отвечай на русском языке."""


MODERATOR_PROMPT = """Ты — модератор фокус-группы на тему "Гибридные автомобили".
Твоя задача — вести дискуссию, задавать вопросы, вовлекать участников.

Сценарий фокус-группы:
1. Раунд 1: Знакомство. Пусть каждый представится и расскажет о своём опыте с автомобилями.
2. Раунд 2: Что участники знают о гибридных автомобилях? Какие ассоциации?
3. Раунд 3: Что нравится и что не нравится в гибридах? (Цена, экология, надёжность, технология)
4. Раунд 4+ (если нужно): Свободное обсуждение, выводы.

Ты должен задавать вопросы, указывая имя участника, к которому обращаешься.
Когда все участники высказались по теме раунда, переходи к следующему раунду.
В конце каждого раунда кратко суммируй ключевые мнения и переходи дальше.
Пиши на русском языке.
"""


def simulate_focus_group(personas: List[Persona], max_rounds: int = 4) -> Transcript:
    """
    Симулирует дискуссию фокус-группы: модератор + персоны.
    Каждый раунд: модератор задаёт вопрос → каждая персона отвечает.
    """
    utterances = []

    # --- Системные промпты для каждой персоны ---
    persona_prompts = {p.id: build_system_prompt(p) for p in personas}
    persona_names = {p.id: p.name for p in personas}

    # --- Сессия: истории сообщений ---
    persona_messages: dict[int, list] = {p.id: [{"role": "system", "content": persona_prompts[p.id]}] for p in personas}
    moderator_messages = [{"role": "system", "content": MODERATOR_PROMPT}]

    # --- История дискуссии для контекста ---
    discussion_history: list[str] = []

    for round_num in range(1, max_rounds + 1):
        # 1. Модератор задаёт вопрос
        context = "\n".join(discussion_history[-20:]) if discussion_history else "Начало дискуссии."
        mod_response = CLIENT.chat.completions.create(
            model=MODEL,
            messages=moderator_messages + [
                {"role": "user", "content": f"Текущий раунд {round_num}. Контекст дискуссии:\n{context}\n\nКакой вопрос задашь группе?"}
            ],
            temperature=0.7,
        )
        moderator_question = mod_response.choices[0].message.content
        moderator_messages.append({"role": "user", "content": f"Раунд {round_num}: задай вопрос"})
        moderator_messages.append({"role": "assistant", "content": moderator_question})
        utterances.append(Utterance(persona_id=0, persona_name="Модератор", text=moderator_question, round=round_num))
        discussion_history.append(f"Модератор (раунд {round_num}): {moderator_question}")

        # 2. Каждая персона отвечает (в случайном порядке для разнообразия)
        import random
        shuffled_ids = list(persona_messages.keys())
        random.shuffle(shuffled_ids)

        for pid in shuffled_ids:
            persona = next(p for p in personas if p.id == pid)
            context = "\n".join(discussion_history[-15:]) if discussion_history else ""
            prompt = f"Раунд {round_num}. Вопрос модератора: {moderator_question}\n\nКонтекст дискуссии:\n{context}\n\nТвой ответ (2-4 предложения, в роли):"
            response = CLIENT.chat.completions.create(
                model=MODEL,
                messages=persona_messages[pid] + [
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
            )
            reply = response.choices[0].message.content
            persona_messages[pid].append({"role": "user", "content": prompt})
            persona_messages[pid].append({"role": "assistant", "content": reply})
            utterances.append(Utterance(persona_id=pid, persona_name=persona_names[pid], text=reply, round=round_num))
            discussion_history.append(f"{persona_names[pid]} (раунд {round_num}): {reply}")

        # 3. Модератор подводит мини-итог раунда
        context = "\n".join(discussion_history[-20:])
        summary_response = CLIENT.chat.completions.create(
            model=MODEL,
            messages=moderator_messages + [
                {"role": "user", "content": f"Подведи краткий итог раунда {round_num} на основе дискуссии:\n{context}"}
            ],
            temperature=0.5,
        )
        round_summary = summary_response.choices[0].message.content
        moderator_messages.append({"role": "user", "content": f"Подведи итог раунда {round_num}"})
        moderator_messages.append({"role": "assistant", "content": round_summary})
        utterances.append(Utterance(persona_id=0, persona_name="Модератор", text=f"[Итог раунда {round_num}]: {round_summary}", round=round_num))
        discussion_history.append(f"Модератор (итог раунда {round_num}): {round_summary}")

    return Transcript(topic="Гибридные автомобили", utterances=utterances)


def save_transcript(transcript: Transcript, path: str = "output/transcript.json"):
    """Сохраняет транскрипт в JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(transcript.model_dump(), f, ensure_ascii=False, indent=2)
    print(f"Транскрипт сохранён в {path}")


def load_transcript(path: str = "output/transcript.json") -> Transcript:
    """Загружает транскрипт из JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Transcript(**data)


if __name__ == "__main__":
    from personas import load_personas
    personas = load_personas()
    transcript = simulate_focus_group(personas, max_rounds=4)
    save_transcript(transcript)
    print(f"Транскрипт: {len(transcript.utterances)} реплик, {transcript.topic}")