"""
Оркестратор: главный цикл Планировщик-Исполнитель-Критик.

Домашнее задание C6:
- validate_plan — валидатор схемы между Планировщиком и Исполнителем
- _topological_levels — уровни вместо плоской сортировки (для параллельности)
- execute_level — параллельный запуск независимых подвопросов через ThreadPoolExecutor
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from planner import planner
from schemas_pwc import Plan, SubQuestion, WorkerAnswer
from worker import worker

VALID_TOOLS = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}


def validate_plan(plan: Plan) -> list[str]:
    """Вернуть список ошибок плана (пустой — всё ок)."""
    errors: list[str] = []
    for sq in plan.subquestions:
        for tool in sq.expected_tools:
            if tool not in VALID_TOOLS:
                errors.append(
                    f"Подвопрос {sq.id}: инструмент '{tool}' не существует. "
                    f"Допустимые: {sorted(VALID_TOOLS)}"
                )
    return errors


def _topological_levels(subqs: list[SubQuestion]) -> list[list[SubQuestion]]:
    """Отсортировать подвопросы по уровням зависимостей.
    
    Возвращает список уровней: на уровне 0 — подвопросы без зависимостей,
    на уровне N — подвопросы, чьи зависимости все на уровнях < N.
    """
    by_id = {s.id: s for s in subqs}
    # Сначала делаем топологическую сортировку DFS (проверка на циклы)
    ordered: list[SubQuestion] = []
    visited: set[int] = set()

    def visit(node_id: int, path: list[int]):
        if node_id in visited:
            return None
        if node_id in path:
            raise ValueError(f"Цикл в depends_on: {path + [node_id]}")
        if node_id not in by_id:
            return None
        for dep in by_id[node_id].depends_on:
            visit(dep, path + [node_id])
        visited.add(node_id)
        ordered.append(by_id[node_id])

    for sq in subqs:
        visit(sq.id, [])

    # Разбиваем ordered на уровни
    levels: list[list[SubQuestion]] = []
    remaining = set(ordered)
    
    while remaining:
        level_ids: set[int] = set()
        for sq in ordered:
            if sq.id not in remaining:
                continue
            # Все зависимости этого подвопроса уже не в remaining?
            deps = set(sq.depends_on)
            if not (deps & remaining):
                level_ids.add(sq.id)
        
        if not level_ids:
            # Остались только узлы с циклическими зависимостями (д.б. отловлено выше)
            break
        
        level = [by_id[i] for i in sorted(level_ids) if i in remaining]
        levels.append(level)
        remaining -= level_ids
    
    return levels


def execute_level(
    level: list[SubQuestion],
    prev_answers: dict[int, WorkerAnswer],
) -> dict[int, WorkerAnswer]:
    """Прогнать все подвопросы уровня параллельно."""
    level_answers: dict[int, WorkerAnswer] = {}
    
    if len(level) <= 1:
        # Для одного подвопроса не создаём пул
        for sq in level:
            level_answers[sq.id] = worker(sq, prev_answers=prev_answers)
        return level_answers
    
    with ThreadPoolExecutor(max_workers=len(level)) as executor:
        future_to_id = {
            executor.submit(worker, sq, prev_answers=prev_answers): sq.id
            for sq in level
        }
        for future in as_completed(future_to_id):
            sq_id = future_to_id[future]
            try:
                level_answers[sq_id] = future.result()
            except Exception as e:
                # В случае ошибки создаём заглушку
                level_answers[sq_id] = WorkerAnswer(
                    subquestion_id=sq_id,
                    question_snippet="(ошибка параллельного исполнения)",
                    answer=f"(ошибка: {e})",
                    used_tools=[],
                    raw_trace=[],
                )
    
    return level_answers


def _synthesize(
    question: str,
    plan: Plan,
    answers: dict[int, WorkerAnswer],
) -> str:
    """Собрать финальный ответ одним LLM-вызовом без tools."""
    parts = [answers[i].answer for i in sorted(answers)]
    return " · ".join(parts)  # заглушка: склейка через точки


def run_pwc(
    question: str, *, max_iter: int = 3, verbose: bool = True,
    use_validator: bool = True,
) -> dict[str, Any]:
    """Запустить цикл Планировщик-Исполнитель-Критик.

    Args:
        question: вопрос пользователя.
        max_iter: макс. число итераций critic-цикла (не путать с повторами валидатора).
        verbose: печатать ли лог.
        use_validator: если True — перед каждым выполнением плана проверять
                       expected_tools на выдуманные инструменты и при обнаружении
                       перезапрашивать план. Если False — пропустить проверку.
    """
    trace: list[dict[str, Any]] = []

    plan = planner(question)
    
    # Валидатор схемы — проверяем план на выдуманные инструменты
    if use_validator:
        for _ in range(max_iter):  # цикл повторных попыток перепланировки
            errors = validate_plan(plan)
            if not errors:
                break
            if verbose:
                print(f"[validator] Найдены несуществующие инструменты: {errors}")
            plan = planner(question, feedback=f"Инструменты не существуют: {errors}")
        else:
            # Если после max_iter попыток всё ещё есть ошибки — сохраняем последний план
            if verbose:
                print(f"[validator] Не удалось исправить план за {max_iter} попыток")
    
    trace.append(
        {
            "iter": 0,
            "kind": "plan",
            "reasoning": plan.reasoning,
            "subquestions": [sq.model_dump() for sq in plan.subquestions],
        }
    )

    if verbose:
        print(f"\n[plan] {plan.reasoning}")
        for sq in plan.subquestions:
            print(f"  {sq.id}. [{','.join(sq.expected_tools)}] {sq.question}")

    for iter_num in range(1, max_iter + 1):
        answers: dict[int, WorkerAnswer] = {}
        levels = _topological_levels(plan.subquestions)
        
        for level_idx, level in enumerate(levels):
            if verbose:
                print(f"  [level {level_idx}] {len(level)} подвопрос(ов)")
            level_answers = execute_level(level, prev_answers=answers)
            answers.update(level_answers)
            
            for sq_id in sorted(level_answers):
                ans = level_answers[sq_id]
                trace.append(
                    {
                        "iter": iter_num,
                        "kind": "worker",
                        "sq_id": sq_id,
                        "used_tools": ans.used_tools,
                        "answer": ans.answer,
                    }
                )
                if verbose:
                    print(f"  [{sq_id}] → {ans.answer}   tools={ans.used_tools}")

        verdict = critic(question, plan, answers)
        trace.append(
            {
                "iter": iter_num,
                "kind": "verdict",
                "ok": verdict.ok,
                "action": verdict.action,
                "reason": verdict.reason,
                "rework_ids": verdict.rework_ids,
            }
        )

        if verbose:
            mark = "✅" if verdict.ok else "❌"
            print(f"  [critic {mark}] {verdict.action}: {verdict.reason}")

        if verdict.ok:
            final = _synthesize(question, plan, answers)
            return {
                "answer": final,
                "plan": plan,
                "answers": answers,
                "trace": trace,
                "iterations": iter_num,
            }

        # Обработка replan/rework
        if verdict.action == "replan":
            plan = planner(question, feedback=verdict.reason)
            # Снова валидируем перепланированный план
            errors = validate_plan(plan)
            if errors:
                if verbose:
                    print(f"[validator] После replan снова выдуманные инструменты: {errors}")
                plan = planner(question, feedback=f"Инструменты не существуют: {errors}")
            continue
        elif verdict.action == "rework":
            # Для rework передаём feedback с указанием id
            feedback = f"{verdict.reason} (переделать подвопросы: {verdict.rework_ids})"
            plan = planner(question, feedback=feedback)
            errors = validate_plan(plan)
            if errors:
                plan = planner(question, feedback=f"Инструменты не существуют: {errors}")
            continue

    return {
        "answer": None,
        "error": f"не удалось получить вердикт 'accept' за {max_iter} итераций",
        "plan": plan,
        "answers": answers,
        "trace": trace,
        "iterations": max_iter,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+", help="Вопрос к агенту")
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--trace", type=Path, default=None, help="Куда сохранить JSON-лог (если задан)"
    )
    args = ap.parse_args()

    q = " ".join(args.query)
    res = run_pwc(q, max_iter=args.max_iter, verbose=not args.quiet)

    print("\n=== ВОПРОС ===")
    print(q)
    print("\n=== ОТВЕТ ===")
    print(res.get("answer") or res.get("error"))
    print(f"\n(итераций: {res.get('iterations', '?')})")

    if args.trace:
        args.trace.write_text(
            json.dumps(
                {"query": q, **_serialize(res)},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Трейс сохранён: {args.trace}")


def _serialize(res: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in res.items():
        if k == "plan" and v is not None:
            out[k] = v.model_dump()
        elif k == "answers":
            out[k] = {i: a.model_dump() for i, a in v.items()}
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()