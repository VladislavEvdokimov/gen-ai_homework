"""
LLM-as-Judge: оценка качества симуляции фокус-группы и проверка галлюцинаций.
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from schema import Transcript, FocusGroupReport, JudgeEvaluation

load_dotenv()

CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o")

JUDGE_PROMPT = """Ты — эксперт по оценке качества симуляций фокус-групп. Оцени результаты симуляции.

У тебя есть:
1. Транскрипт дискуссии (реплики участников и модератора)
2. Итоговый отчёт (сводки по аспектам, общий сентимент, рекомендации)

Оцени по следующим критериям (каждый от 1 до 5):

1. persona_consistency_score — насколько каждая персона придерживалась своей роли (возраст, профессия, доход, отношение к гибридам)?
2. hallucination_count — сколько раз в отчёте встречаются ghost-цитаты или факты, которых нет в транскрипте? (число >= 0)
3. hallucination_examples — примеры галлюцинаций (если есть)
4. transcript_grounding_score — насколько выводы в отчёте привязаны к конкретным репликам из транскрипта?
5. overall_quality_score — общее качество симуляции (реалистичность, глубина, связность)

Верни JSON:
{
  "persona_consistency_score": число 1-5,
  "hallucination_count": число >= 0,
  "hallucination_examples": ["пример 1", "пример 2"],
  "transcript_grounding_score": число 1-5,
  "overall_quality_score": число 1-5,
  "comments": "развёрнутый комментарий с замечаниями"
}
"""


def evaluate(transcript: Transcript, report: FocusGroupReport) -> JudgeEvaluation:
    """
    Оценивает качество симуляции с помощью LLM-судьи.
    """
    # Форматируем транскрипт (сокращённо, чтобы влезло в контекст)
    transcript_lines = []
    for utt in transcript.utterances[:50]:  # первые 50 реплик для экономии токенов
        speaker = "Модератор" if utt.persona_id == 0 else utt.persona_name
        transcript_lines.append(f"[Раунд {utt.round}] {speaker}: {utt.text[:200]}")

    transcript_text = "\n".join(transcript_lines)
    report_text = json.dumps(report.model_dump(), ensure_ascii=False, indent=2)

    response = CLIENT.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": f"=== ТРАНСКРИПТ ===\n{transcript_text}\n\n=== ОТЧЁТ ===\n{report_text}\n\nОцени качество симуляции."
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)

    judge = JudgeEvaluation(
        persona_consistency_score=data.get("persona_consistency_score", 3),
        hallucination_count=data.get("hallucination_count", 0),
        hallucination_examples=data.get("hallucination_examples", []),
        transcript_grounding_score=data.get("transcript_grounding_score", 3),
        overall_quality_score=data.get("overall_quality_score", 3),
        comments=data.get("comments", ""),
    )

    return judge


def save_judge_result(judge: JudgeEvaluation, path: str = "output/judge_report.json"):
    """Сохраняет результат оценки в JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(judge.model_dump(), f, ensure_ascii=False, indent=2)
    print(f"Оценка Judge сохранена в {path}")
    print(f"  Согласованность персон: {judge.persona_consistency_score}/5")
    print(f"  Галлюцинаций: {judge.hallucination_count}")
    print(f"  Привязка к транскрипту: {judge.transcript_grounding_score}/5")
    print(f"  Общее качество: {judge.overall_quality_score}/5")


def load_judge_result(path: str = "output/judge_report.json") -> JudgeEvaluation:
    """Загружает результат оценки из JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JudgeEvaluation(**data)


if __name__ == "__main__":
    from simulation import load_transcript
    from map_reduce import load_report
    transcript = load_transcript()
    report = load_report()
    judge = evaluate(transcript, report)
    save_judge_result(judge)