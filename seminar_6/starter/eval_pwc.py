"""
Eval мульти-агента: 6 вопросов × 3 конфигурации.

Три конфигурации:
  1. Одиночный агент С5 (agent_s5.run_agent)
  2. PWC без валидатора (orchestrator.run_pwc)
  3. PWC + валидатор (orchestrator.run_pwc — валидатор встроен в run_pwc)

Прогон N=5 раз, считаем долю успешных прогонов.

Запуск:
    python eval_pwc.py           # полный прогон
    python eval_pwc.py --single  # только один прогон каждого, быстрая проверка
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_s5 import run_agent
from orchestrator import run_pwc


CASES = [
    {
        "id": "Q1",
        "query": "Во сколько раз USD подорожал с 1 января 2022 по сегодня?",
        "comment": (
            "Класс ошибки C: одиночный часто считает в уме, не зовёт calculate. "
            "PWC должен починить — Планировщик обязан добавить calculate-подвопрос."
        ),
        "must_have_keywords": ["раз", "USD"],
    },
    {
        "id": "Q2",
        "query": (
            "Какая сейчас реальная ключевая ставка, если инфляцию брать "
            "по последнему доступному месяцу, а не по году?"
        ),
        "comment": (
            "Класс ошибки B: одиночный не умеет искать «последний доступный» "
            "месяц, зацикливается. PWC должен разбить на шаги."
        ),
        "must_have_keywords": ["%"],
    },
    {
        "id": "Q3",
        "query": (
            "Какова накопленная инфляция с января 2022 по март 2026? "
            "Рассчитай как произведение всех (1 + ипц_м/100) по месяцам."
        ),
        "comment": (
            "Класс ошибки D (граница паттерна): требует get_inflation за много "
            "месяцев + большое calculate-выражение. Одиночный галлюцинирует "
            "get_cumulative_inflation; PWC обычно тоже (Планировщик может добавить "
            "выдуманный инструмент в план). Валидатор должен поймать это."
        ),
        "must_have_keywords": ["%"],
    },
    # Q4: Гарантированно чинится валидатором (одиночный и PWC падают, PWC+валятор справл.)
    {
        "id": "Q4",
        "query": (
            "Какова накопленная инфляция за 2023 год? "
            "Рассчитай как произведение всех (1 + ипц_м/100) по месяцам."
        ),
        "comment": (
            "Одиночный агент и PWC без валидатора: Планировщик выдумывает "
            "get_cumulative_inflation → fail. PWC+валидатор: ловит выдумку, "
            "Планировщик перепланирует с 12× get_inflation + calculate → OK."
        ),
        "must_have_keywords": ["%"],
    },
    # Q5: Естественная параллельность (3 независимых подвопроса)
    {
        "id": "Q5",
        "query": (
            "Сравни текущие курсы USD, EUR и CNY к рублю. "
            "Какая валюта самая дорогая, а какая самая дешёвая?"
        ),
        "comment": (
            "Три независимых подвопроса (depends_on=[] у всех). "
            "Идеальный кейс для параллельного исполнения."
        ),
        "must_have_keywords": ["USD", "EUR", "CNY"],
    },
    # Q6: Реальный макро-вопрос
    {
        "id": "Q6",
        "query": (
            "Какая была реальная ключевая ставка в России на начало "
            "2024 года (январь), если считать реальную ставку как "
            "номинальная ставка минус инфляция г/г за январь 2024?"
        ),
        "comment": (
            "Требует get_key_rate на дату, get_inflation за январь 2024 "
            "и calculate. Проверяет декомпозицию с привязкой к дате."
        ),
        "must_have_keywords": ["%"],
    },
]


VALID_TOOL_NAMES = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}


def _check_single(case: dict, result: dict) -> dict:
    """Проверить результат одиночного прогона."""
    used = {e["call"] for e in result.get("trace", []) if "call" in e}
    ans = (result.get("answer") or "").lower()
    hallucinated = used - VALID_TOOL_NAMES
    must = all(kw.lower() in ans for kw in case["must_have_keywords"]) if case["must_have_keywords"] else True
    arith_without_calc = (
        case["id"] in {"Q1", "Q2", "Q3", "Q6"}
        and "calculate" not in used
        and bool(ans)
    )
    ok = bool(ans) and not hallucinated and must and not arith_without_calc
    return {
        "ok": ok,
        "used_tools": sorted(used),
        "hallucinated": sorted(hallucinated),
        "must_have_ok": must,
        "arith_without_calc": arith_without_calc,
        "answer_preview": (result.get("answer") or "")[:180],
    }


def _check_pwc(case: dict, result: dict) -> dict:
    """Проверить результат PWC-прогона."""
    used = set()
    for t in result.get("trace", []):
        if t.get("kind") == "worker":
            used.update(t.get("used_tools") or [])
    ans = (result.get("answer") or "").lower()
    hallucinated = used - VALID_TOOL_NAMES
    # Также проверим галлюцинации на этапе Планировщика (в плане expected_tools)
    plan_tools = set()
    plan = result.get("plan")
    if plan is not None:
        for sq in plan.subquestions:
            plan_tools.update(sq.expected_tools)
    plan_hallucinated = plan_tools - VALID_TOOL_NAMES

    must = all(kw.lower() in ans for kw in case["must_have_keywords"]) if case["must_have_keywords"] else True
    ok = (
        bool(result.get("answer"))
        and not hallucinated
        and not plan_hallucinated
        and must
    )
    return {
        "ok": ok,
        "used_tools": sorted(used),
        "plan_tools": sorted(plan_tools),
        "hallucinated_in_workers": sorted(hallucinated),
        "hallucinated_in_plan": sorted(plan_hallucinated),
        "must_have_ok": must,
        "iterations": result.get("iterations", -1),
        "answer_preview": (result.get("answer") or "")[:180],
    }


def run_case(case: dict, *, n: int = 5) -> dict:
    single = {"runs": [], "pass": 0}
    pwc = {"runs": [], "pass": 0}
    pwc_validator = {"runs": [], "pass": 0}

    for i in range(n):
        # --- Одиночный агент ---
        try:
            r1 = run_agent(case["query"], max_iter=8, verbose=False)
        except Exception as e:
            r1 = {"answer": None, "error": f"{type(e).__name__}: {e}", "trace": []}
        check1 = _check_single(case, r1)
        single["runs"].append(check1)
        single["pass"] += int(check1["ok"])

        # --- PWC без валидатора ---
        try:
            r2 = run_pwc(case["query"], max_iter=3, verbose=False, use_validator=False)
        except Exception as e:
            r2 = {"answer": None, "error": f"{type(e).__name__}: {e}",
                  "trace": [], "plan": None}
        check2 = _check_pwc(case, r2)
        pwc["runs"].append(check2)
        pwc["pass"] += int(check2["ok"])

        # --- PWC + валидатор ---
        try:
            r3 = run_pwc(case["query"], max_iter=3, verbose=False, use_validator=True)
        except Exception as e:
            r3 = {"answer": None, "error": f"{type(e).__name__}: {e}",
                  "trace": [], "plan": None}
        check3 = _check_pwc(case, r3)
        pwc_validator["runs"].append(check3)
        pwc_validator["pass"] += int(check3["ok"])

    return {
        "id": case["id"],
        "query": case["query"],
        "comment": case["comment"],
        "n": n,
        "single": single,
        "pwc": pwc,
        "pwc_validator": pwc_validator,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true",
                    help="Только один прогон каждого кейса (быстро)")
    ap.add_argument("-n", type=int, default=5,
                    help="Сколько прогонов на кейс (default=5)")
    args = ap.parse_args()
    n = 1 if args.single else args.n

    print(f"Eval С6: {len(CASES)} кейсов × {n} прогонов\n")
    results = []
    for case in CASES:
        print(f"=== {case['id']}: {case['query'][:70]}...")
        r = run_case(case, n=n)
        results.append(r)
        s = r["single"]; p = r["pwc"]; pv = r["pwc_validator"]
        print(f"   single: {s['pass']}/{n}    pwc: {p['pass']}/{n}    pwc+validator: {pv['pass']}/{n}")
        for run in p["runs"][:1]:
            if run["hallucinated_in_plan"]:
                print(f"   ⚠ План содержит выдуманные инструменты: {run['hallucinated_in_plan']}")
        print()

    # Итог
    print("=" * 70)
    print("ИТОГО:")
    print(f"{'id':<6} {'single':>8} {'pwc':>8} {'pwc+val':>8}  query")
    print("-" * 70)
    for r in results:
        print(f"  {r['id']:<4} {r['single']['pass']:>3}/{n:1}   "
              f"{r['pwc']['pass']:>3}/{n:1}     "
              f"{r['pwc_validator']['pass']:>3}/{n:1}   — {r['query'][:60]}")

    out = Path(__file__).parent / "eval_pwc_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2,
                              default=str), encoding="utf-8")
    print(f"\nРезультаты: {out}")


if __name__ == "__main__":
    main()