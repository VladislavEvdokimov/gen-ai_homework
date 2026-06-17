"""
Eval: запуск пайплайна на наборе тестовых входов и сбор метрик.
Требование: >=15 тестовых входов.
Сохраняет результаты после каждого теста для устойчивости к таймаутам.
"""

import json
import os
import sys
import time
import csv
from typing import List, Dict, Optional
from dotenv import load_dotenv
from schema import EvalCase
from pipeline import run_pipeline, save_pipeline_stats

load_dotenv()


# 3 темы x 5 seed-значений = 15 комбинаций
TOPIC_VARIANTS = [
    "Гибридные автомобили",
    "Экономия топлива в гибридах",
    "Экологичность гибридов",
]

PERSONA_SEEDS = [42, 73, 101, 204, 307]

MAX_ROUNDS = 3  # 3 раунда для экономии времени на eval

CACHE_FILE = "output/eval_cache.json"


def check_passed(judge_result) -> bool:
    """Критерий прохождения теста."""
    if judge_result is None:
        return False
    return (
        judge_result.persona_consistency_score >= 3
        and judge_result.hallucination_count <= 2
        and judge_result.transcript_grounding_score >= 3
    )


def load_cache() -> dict:
    """Загружает кеш выполненных тестов."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    """Сохраняет кеш."""
    os.makedirs("output", exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def make_cache_key(topic: str, seed: int) -> str:
    return f"{topic}:::{seed}"


def run_eval(force_regen: bool = False) -> List[EvalCase]:
    """
    Запускает eval на 15 комбинациях (3 темы x 5 seed-значений).
    Сохраняет результаты после каждого теста.
    """
    cases_path = "output/eval_cases.json"

    cache = load_cache()
    cases = []

    total = len(TOPIC_VARIANTS) * len(PERSONA_SEEDS)
    print(f"Запуск eval на {total} комбинациях...")
    if cache:
        print(f"  Найдено {len(cache)} кешированных тестов")
    print()

    case_id = 0

    for topic in TOPIC_VARIANTS:
        for seed in PERSONA_SEEDS:
            case_id += 1
            cache_key = make_cache_key(topic, seed)

            # Проверка кеша
            if not force_regen and cache_key in cache:
                print(f"  [CACHE] Тест {case_id}/{total}: topic='{topic}', seed={seed}")
                case = EvalCase(**cache[cache_key])
                cases.append(case)
                status = "[OK]" if case.passed else "[FAIL]"
                print(f"  {status} Загружен из кеша")
                continue

            print(f"\n{'='*60}")
            print(f"Тест {case_id}/{total}: topic='{topic}', seed={seed}")
            print(f"{'='*60}")

            start = time.time()
            error = None
            result_data = None

            try:
                result = run_pipeline(
                    persona_seed=seed,
                    max_rounds=MAX_ROUNDS,
                    topic_variant=topic,
                )
                result_data = result
                passed = check_passed(result.get("judge_result"))
            except Exception as e:
                error = str(e)
                passed = False
                print(f"  [ERR] {error}")

            duration = time.time() - start

            case = EvalCase(
                case_id=case_id,
                persona_seed=seed,
                topic_variant=topic,
                transcript=result_data.get("transcript") if result_data else None,
                entities=result_data.get("entities") if result_data else None,
                report=result_data.get("report") if result_data else None,
                judge=result_data.get("judge_result") if result_data else None,
                cost=result_data["stats"]["total_cost"] if result_data and result_data.get("stats") else 0.0,
                steps=result_data["stats"]["steps"] if result_data and result_data.get("stats") else 0,
                tokens=result_data["stats"]["total_tokens"] if result_data and result_data.get("stats") else 0,
                passed=passed,
                error=error,
            )
            cases.append(case)

            # Сохраняем в кеш
            cache[cache_key] = case.model_dump()
            save_cache(cache)

            status = "[OK]" if passed else "[FAIL]"
            cost_str = f"${case.cost:.6f}" if case.cost > 0 else "N/A"
            print(f"  {status} {'PASS' if passed else 'FAIL'}")
            print(f"  Время: {duration:.1f}с | Цена: {cost_str} | {case.tokens} токенов")
            if case.judge:
                print(f"  Согласованность: {case.judge.persona_consistency_score}/5, "
                      f"Галлюцинации: {case.judge.hallucination_count}, "
                      f"Привязка: {case.judge.transcript_grounding_score}/5")

    # Финальное сохранение
    save_eval_results(cases, "output/eval_results.csv", cases_path)
    return cases


def save_eval_results(cases: List[EvalCase], csv_path: str, json_path: str):
    """Сохраняет результаты eval в CSV и JSON."""
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id", "topic", "seed", "passed", "steps", "tokens", "cost",
            "persona_consistency", "hallucination_count", "transcript_grounding",
            "overall_quality", "error"
        ])
        for c in cases:
            writer.writerow([
                c.case_id,
                c.topic_variant,
                c.persona_seed,
                c.passed,
                c.steps,
                c.tokens,
                f"{c.cost:.6f}",
                c.judge.persona_consistency_score if c.judge else "",
                c.judge.hallucination_count if c.judge else "",
                c.judge.transcript_grounding_score if c.judge else "",
                c.judge.overall_quality_score if c.judge else "",
                c.error or "",
            ])
    print(f"\nРезультаты eval сохранены в {csv_path}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in cases], f, ensure_ascii=False, indent=2)
    print(f"Результаты eval сохранены в {json_path}")


def print_summary(cases: List[EvalCase]):
    """Печатает сводку результатов eval."""
    total = len(cases)
    passed = sum(1 for c in cases if c.passed)
    failed = total - passed

    print("\n" + "=" * 60)
    print("ИТОГОВАЯ СВОДКА EVAL")
    print("=" * 60)
    print(f"Всего тестов: {total}")
    print(f"Пройдено: {passed} ({passed / total * 100:.1f}%)")
    print(f"Провалено: {failed} ({failed / total * 100:.1f}%)")
    print()

    if cases:
        avg_cost = sum(c.cost for c in cases) / total
        avg_tokens = sum(c.tokens for c in cases) / total
        avg_steps = sum(c.steps for c in cases) / total
        print(f"Средняя стоимость: ${avg_cost:.6f}")
        print(f"Среднее число токенов: {avg_tokens:.0f}")
        print(f"Среднее число шагов: {avg_steps:.1f}")

        if any(c.judge for c in cases):
            valid = [c for c in cases if c.judge]
            avg_consistency = sum(c.judge.persona_consistency_score for c in valid) / len(valid)
            avg_grounding = sum(c.judge.transcript_grounding_score for c in valid) / len(valid)
            avg_quality = sum(c.judge.overall_quality_score for c in valid) / len(valid)
            avg_hallucinations = sum(c.judge.hallucination_count for c in valid) / len(valid)
            print(f"\nСредняя согласованность персон: {avg_consistency:.1f}/5")
            print(f"Средняя привязка к транскрипту: {avg_grounding:.1f}/5")
            print(f"Среднее общее качество: {avg_quality:.1f}/5")
            print(f"Среднее число галлюцинаций: {avg_hallucinations:.1f}")

        print(f"\nПровалы:")
        for c in cases:
            if not c.passed and c.error:
                print(f"  [FAIL] Тест {c.case_id} ({c.topic_variant}, seed={c.persona_seed}): {c.error[:100]}")
            elif not c.passed:
                reason_parts = []
                if c.judge:
                    if c.judge.persona_consistency_score < 3:
                        reason_parts.append(f"согласованность={c.judge.persona_consistency_score}")
                    if c.judge.hallucination_count > 2:
                        reason_parts.append(f"галлюцинации={c.judge.hallucination_count}")
                    if c.judge.transcript_grounding_score < 3:
                        reason_parts.append(f"привязка={c.judge.transcript_grounding_score}")
                reason = ", ".join(reason_parts) if reason_parts else "неизвестно"
                print(f"  [FAIL] Тест {c.case_id} ({c.topic_variant}, seed={c.persona_seed}): {reason}")

    print("=" * 60)


if __name__ == "__main__":
    force = "--force" in sys.argv
    cases = run_eval(force_regen=force)
    print_summary(cases)