"""
Главный пайплайн: симуляция фокус-группы + анализ.
"""

import os
import json
import time
import tiktoken
from dotenv import load_dotenv
from personas import generate_personas, save_personas
from simulation import simulate_focus_group, save_transcript
from ie_extraction import extract_entities, save_entities
from aspect_analysis import analyze_aspects, save_aspect_analysis
from map_reduce import map_reduce_summary, save_report
from judge import evaluate, save_judge_result
from schema import Persona, Transcript, Entity, AspectAnalysis, FocusGroupReport, JudgeEvaluation

load_dotenv()


class PipelineStats:
    """Статистика выполнения пайплайна."""
    def __init__(self):
        self.steps = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.start_time = 0.0
        self.end_time = 0.0

    def add_request(self, prompt_tokens: int, completion_tokens: int, model: str):
        """Учитывает токены и стоимость запроса."""
        self.total_tokens += prompt_tokens + completion_tokens
        # Цены GPT-4o-mini: $0.15 / 1M input, $0.60 / 1M output
        # GPT-4o: $2.50 / 1M input, $10.00 / 1M output
        if "mini" in model:
            input_cost = prompt_tokens * 0.15 / 1_000_000
            output_cost = completion_tokens * 0.60 / 1_000_000
        else:
            input_cost = prompt_tokens * 2.50 / 1_000_000
            output_cost = completion_tokens * 10.00 / 1_000_000
        self.total_cost += input_cost + output_cost

    def add_step(self):
        self.steps += 1

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Приблизительная оценка количества токенов."""
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return len(text) // 2  # грубая оценка


def run_pipeline(
    persona_seed: int = 42,
    max_rounds: int = 4,
    topic_variant: str = "Гибридные автомобили",
) -> dict:
    """
    Запускает полный пайплайн: генерация персон → симуляция → IE → аспекты → Map-Reduce → Judge.

    Возвращает словарь со всеми артефактами и статистикой.
    """
    stats = PipelineStats()
    stats.start_time = time.time()

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    judge_model = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o")

    print("=" * 60)
    print(f"Пайплайн: Симуляция фокус-группы")
    print(f"Тема: {topic_variant}")
    print(f"Seed персон: {persona_seed}")
    print(f"Раундов: {max_rounds}")
    print("=" * 60)

    # Шаг 1: Генерация персон
    print("\n[1/5] Генерация синтетических персон...")
    personas = generate_personas(seed=persona_seed)
    save_personas(personas)
    stats.add_step()
    print(f"  Сгенерировано {len(personas)} персон")

    # Шаг 2: Симуляция дискуссии
    print("\n[2/5] Симуляция фокус-группы...")
    transcript = simulate_focus_group(personas, max_rounds=max_rounds)
    save_transcript(transcript)
    stats.add_step()
    print(f"  Транскрипт: {len(transcript.utterances)} реплик")
    # Учитываем токены симуляции (приблизительно)
    for utt in transcript.utterances:
        stats.add_request(
            prompt_tokens=estimate_tokens(utt.text) // 2,
            completion_tokens=estimate_tokens(utt.text) // 2,
            model=model,
        )

    # Шаг 3: IE — извлечение сущностей
    print("\n[3/5] Извлечение структурированных сущностей (IE)...")
    entities = extract_entities(transcript)
    save_entities(entities)
    stats.add_step()
    print(f"  Извлечено сущностей: {len(entities)}")

    # Шаг 4: Аспектный анализ + Map-Reduce
    print("\n[4/5] Аспектный анализ...")
    aspect_analysis = analyze_aspects(entities)
    save_aspect_analysis(aspect_analysis)
    stats.add_step()
    print(f"  Групп аспектов: {len(aspect_analysis.groups)}")

    print("\n[4/5] Map-Reduce суммаризация...")
    report = map_reduce_summary(aspect_analysis.groups)
    save_report(report)
    stats.add_step()
    print(f"  Аспектов в отчёте: {len(report.summaries)}")

    # Шаг 5: LLM-as-Judge
    print("\n[5/5] Оценка качества (LLM-as-Judge)...")
    judge_result = evaluate(transcript, report)
    save_judge_result(judge_result)
    stats.add_step()
    print(f"  Согласованность: {judge_result.persona_consistency_score}/5")
    print(f"  Галлюцинаций: {judge_result.hallucination_count}")
    print(f"  Общее качество: {judge_result.overall_quality_score}/5")

    stats.end_time = time.time()

    # Сводка
    print("\n" + "=" * 60)
    print("СТАТИСТИКА ПАЙПЛАЙНА")
    print(f"  Шагов: {stats.steps}")
    print(f"  Токенов (оценка): ~{stats.total_tokens:,}")
    print(f"  Стоимость (оценка): ${stats.total_cost:.6f}")
    print(f"  Длительность: {stats.duration:.1f} сек")
    print("=" * 60)

    return {
        "personas": personas,
        "transcript": transcript,
        "entities": entities,
        "aspect_analysis": aspect_analysis,
        "report": report,
        "judge_result": judge_result,
        "stats": {
            "steps": stats.steps,
            "total_tokens": stats.total_tokens,
            "total_cost": round(stats.total_cost, 6),
            "duration_seconds": round(stats.duration, 1),
        }
    }


def save_pipeline_stats(stats: dict, path: str = "output/trace.json"):
    """Сохраняет трейс выполнения в JSON."""
    output = {
        "steps": stats["steps"],
        "total_tokens": stats["total_tokens"],
        "total_cost_usd": stats["total_cost"],
        "duration_seconds": stats["duration_seconds"],
        "artifacts": [
            "input/personas.json",
            "output/transcript.json",
            "output/entities.json",
            "output/aspects.json",
            "output/summary.json",
            "output/judge_report.json",
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Трейс сохранён в {path}")


if __name__ == "__main__":
    import sys

    persona_seed = 42
    max_rounds = 4
    topic = "Гибридные автомобили"

    if len(sys.argv) > 1:
        persona_seed = int(sys.argv[1])
    if len(sys.argv) > 2:
        max_rounds = int(sys.argv[2])

    result = run_pipeline(
        persona_seed=persona_seed,
        max_rounds=max_rounds,
        topic_variant=topic,
    )
    save_pipeline_stats(result["stats"])
    print("\nПайплайн завершён успешно!")