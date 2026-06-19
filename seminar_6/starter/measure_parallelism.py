"""
Замер ускорения параллельного исполнения.

Сравнивает время последовательного и параллельного прогона для Q1 (и Q5).
Использует _topological_levels + execute_level (параллельно) против
последовательного выполнения _topological_sort + цикл по одному.

Запуск:
    python measure_parallelism.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent))

from planner import planner
from orchestrator import _topological_levels, execute_level
from schemas_pwc import Plan, SubQuestion, WorkerAnswer
from worker import worker

# Вопросы для замера: Q1 (2 подвопроса с зависимостью) и Q5 (3 независимых)
TEST_QUERIES = [
    {
        "name": "Q1: рост USD (2 подвопроса, зависимость)",
        "query": "Во сколько раз USD подорожал с 1 января 2022 по сегодня?",
    },
    {
        "name": "Q5: курсы USD/EUR/CNY (3 независимых)",
        "query": "Сравни текущие курсы USD, EUR и CNY к рублю. Какая валюта самая дорогая, а какая самая дешёвая?",
    },
]


def run_sequential(plan: Plan) -> dict[int, WorkerAnswer]:
    """Последовательный прогон подвопросов (по уровню, но без параллельности)."""
    levels = _topological_levels(plan.subquestions)
    answers: dict[int, WorkerAnswer] = {}
    for level in levels:
        for sq in level:
            answers[sq.id] = worker(sq, prev_answers=answers)
    return answers


def run_parallel(plan: Plan) -> dict[int, WorkerAnswer]:
    """Параллельный прогон подвопросов (уровнями)."""
    levels = _topological_levels(plan.subquestions)
    answers: dict[int, WorkerAnswer] = {}
    for level in levels:
        level_answers = execute_level(level, prev_answers=answers)
        answers.update(level_answers)
    return answers


def measure(query: str, *, n: int = 5) -> dict:
    """Замерить время последовательного и параллельного прогона, N раз."""
    # Получаем план один раз
    plan = planner(query)
    levels = _topological_levels(plan.subquestions)
    total_sqs = sum(len(level) for level in levels)
    
    seq_times = []
    par_times = []
    
    for i in range(n):
        # Последовательный
        t0 = time.perf_counter()
        run_sequential(plan)
        t1 = time.perf_counter()
        seq_times.append(t1 - t0)
        
        # Параллельный
        t0 = time.perf_counter()
        run_parallel(plan)
        t1 = time.perf_counter()
        par_times.append(t1 - t0)
    
    avg_seq = sum(seq_times) / n
    avg_par = sum(par_times) / n
    speedup = avg_seq / avg_par if avg_par > 0 else float('inf')
    
    return {
        "total_subquestions": total_sqs,
        "levels": len(levels),
        "n": n,
        "seq_times": seq_times,
        "par_times": par_times,
        "avg_seq": avg_seq,
        "avg_par": avg_par,
        "speedup": speedup,
    }


def main():
    print("=" * 70)
    print("ЗАМЕР ПАРАЛЛЕЛЬНОСТИ")
    print("=" * 70)
    
    for test in TEST_QUERIES:
        print(f"\n--- {test['name']} ---")
        print(f"Вопрос: {test['query']}")
        
        result = measure(test["query"], n=5)
        
        print(f"\nПодвопросов: {result['total_subquestions']}, уровней: {result['levels']}")
        print(f"Замеров: {result['n']}")
        print(f"\nПоследовательный (среднее): {result['avg_seq']:.3f} с")
        print(f"Параллельный (среднее):     {result['avg_par']:.3f} с")
        print(f"Ускорение:                  {result['speedup']:.2f}x")
        
        if result["total_subquestions"] > 1 and result["levels"] > 0:
            ideal = min(
                len(l) for l in _topological_levels(
                    planner(test["query"]).subquestions
                )
            )
    
    # Сводка
    print("\n" + "=" * 70)
    print("СВОДНАЯ ТАБЛИЦА:")
    print(f"{'Вопрос':<40} {'Подвопросы':>12} {'Уровни':>8} {'Послед., с':>12} {'Паралл., с':>12} {'Ускорение':>10}")
    print("-" * 94)
    for test in TEST_QUERIES:
        result = measure(test["query"], n=5)
        print(f"{test['name']:<40} {result['total_subquestions']:>12} {result['levels']:>8} "
              f"{result['avg_seq']:>8.3f}   {result['avg_par']:>8.3f}   {result['speedup']:>7.2f}x")


if __name__ == "__main__":
    main()