"""
Map-Reduce: суммаризация мнений по каждому аспекту.
Map: извлекаем ключевые тезисы из каждой группы аспектов.
Reduce: объединяем тезисы в итоговое резюме по аспекту.
"""

import json
import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
from schema import AspectGroup, AspectSummary, FocusGroupReport, Sentiment

load_dotenv()

CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

MAP_PROMPT = """Ты — аналитик. У тебя есть группа мнений участников фокус-группы об аспекте "{aspect}" гибридных автомобилей.

Каждое мнение содержит:
- Кто высказался (persona_name)
- Что сказал (opinion)
- Тональность (sentiment: positive/neutral/negative)
- Прямая цитата (quote)

Твоя задача — извлечь 2-3 ключевых тезиса из этих мнений.
Для каждого тезиса укажи, согласны ли участники (consensus), мнения разделились (mixed) или есть явное несогласие (disagreement).

Верни JSON:
{{
  "theses": [
    {{ "thesis": "текст тезиса", "consensus": "consensus|mixed|disagreement", "supporters": ["имя1", "имя2"] }}
  ]
}}
"""

REDUCE_PROMPT = """Ты — ведущий аналитик. У тебя есть ключевые тезисы по аспекту "{aspect}" гибридных автомобилей, полученные из фокус-группы.

Напиши итоговое резюме (3-5 предложений), которое:
1. Отражает общую картину мнений по этому аспекту
2. Указывает, есть ли консенсус или разногласия
3. Выделяет ключевые проблемы/тезисы

Также определи общий консенсус по аспекту: "consensus" (все согласны), "mixed" (мнения разделились), "disagreement" (явный конфликт мнений).

Верни JSON:
{{
  "summary": "итоговое резюме",
  "consensus": "consensus|mixed|disagreement",
  "key_concerns": ["тезис 1", "тезис 2", "тезис 3"]
}}
"""


def map_aspect(aspect: str, opinions_text: str) -> Dict:
    """Map: извлекает ключевые тезисы из мнений по аспекту."""
    response = CLIENT.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": MAP_PROMPT.format(aspect=aspect)},
            {"role": "user", "content": f"Мнения по аспекту '{aspect}':\n\n{opinions_text}"}
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON decode error in map: {e}")
        print(f"  Raw (first 200): {raw[:200]}")
        # Пытаемся извлечь JSON из ответа
        import re
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        # Возвращаем пустой результат
        return {"theses": []}


def reduce_aspect(aspect: str, theses_text: str) -> Dict:
    """Reduce: объединяет тезисы в итоговое резюме."""
    response = CLIENT.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": REDUCE_PROMPT.format(aspect=aspect)},
            {"role": "user", "content": f"Тезисы по аспекту '{aspect}':\n\n{theses_text}"}
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


def map_reduce_summary(aspect_groups: List[AspectGroup]) -> FocusGroupReport:
    """
    Применяет Map-Reduce ко всем группам аспектов.
    """
    summaries = []

    for group in aspect_groups:
        aspect = group.aspect
        # Форматируем мнения для Map-шага
        opinions_lines = []
        for e in group.opinions:
            opinions_lines.append(
                f"- {e.persona_name} ({e.sentiment.value}): {e.opinion}\n"
                f"  Цитата: \"{e.quote}\""
            )
        opinions_text = "\n".join(opinions_lines)

        print(f"  [Map] Обработка аспекта: {aspect}...")
        map_result = map_aspect(aspect, opinions_text)
        theses = map_result.get("theses", [])
        theses_text = json.dumps(theses, ensure_ascii=False, indent=2)

        print(f"  [Reduce] Суммаризация аспекта: {aspect}...")
        reduce_result = reduce_aspect(aspect, theses_text)

        summary = AspectSummary(
            aspect=aspect,
            summary=reduce_result.get("summary", ""),
            consensus=reduce_result.get("consensus", "mixed"),
            key_concerns=reduce_result.get("key_concerns", []),
        )
        summaries.append(summary)

    # Определяем общий сентимент
    overall = determine_overall_sentiment(aspect_groups)

    # Генерируем рекомендации
    recommendations = generate_recommendations(summaries)

    return FocusGroupReport(
        topic="Гибридные автомобили",
        summaries=summaries,
        overall_sentiment=overall,
        recommendations=recommendations,
    )


def determine_overall_sentiment(aspect_groups: List[AspectGroup]) -> Sentiment:
    """Определяет общий сентимент по всем группам."""
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for group in aspect_groups:
        for e in group.opinions:
            sentiment_counts[e.sentiment.value] += 1

    if sentiment_counts["positive"] > sentiment_counts["negative"]:
        return Sentiment.POSITIVE
    elif sentiment_counts["negative"] > sentiment_counts["positive"]:
        return Sentiment.NEGATIVE
    else:
        return Sentiment.NEUTRAL


def generate_recommendations(summaries: List[AspectSummary]) -> List[str]:
    """Генерирует рекомендации на основе сводок."""
    recommendations = []
    for s in summaries:
        if s.consensus == "disagreement" or s.consensus == "mixed":
            recommendations.append(f"По аспекту «{s.aspect}» мнения разделились — требуется дополнительное исследование.")
        for concern in s.key_concerns[:2]:
            recommendations.append(f"Учтено: {concern}")
    return recommendations[:5]


def save_report(report: FocusGroupReport, path: str = "output/summary.json"):
    """Сохраняет отчёт в JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, ensure_ascii=False, indent=2)
    print(f"Итоговый отчёт сохранён в {path}")
    print(f"  Аспектов: {len(report.summaries)}")
    print(f"  Общий сентимент: {report.overall_sentiment.value}")
    print(f"  Рекомендаций: {len(report.recommendations)}")


def load_report(path: str = "output/summary.json") -> FocusGroupReport:
    """Загружает отчёт из JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return FocusGroupReport(**data)


if __name__ == "__main__":
    from aspect_analysis import load_aspect_analysis
    analysis = load_aspect_analysis()
    report = map_reduce_summary(analysis.groups)
    save_report(report)