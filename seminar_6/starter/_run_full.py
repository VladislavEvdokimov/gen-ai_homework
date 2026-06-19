"""Полный прогон всех скриптов с записью результатов"""
import sys, json, time, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 1. Замер параллельности
print("=" * 60, file=sys.stderr)
print("1. ЗАМЕР ПАРАЛЛЕЛЬНОСТИ", file=sys.stderr)
print("=" * 60, file=sys.stderr)

from measure_parallelism import TEST_QUERIES, measure as par_measure

for test in TEST_QUERIES:
    print(f"  {test['name']}...", file=sys.stderr)
    r = par_measure(test["query"], n=3)
    print(f"    Подвопросов: {r['total_subquestions']}, уровней: {r['levels']}", file=sys.stderr)
    print(f"    Послед.: {r['avg_seq']:.3f}с, Паралл.: {r['avg_par']:.3f}с, Ускорение: {r['speedup']:.2f}x", file=sys.stderr)

# 2. Замер критики
print(file=sys.stderr)
print("=" * 60, file=sys.stderr)
print("2. ЗАМЕР КРИТИКИ", file=sys.stderr)
print("=" * 60, file=sys.stderr)

from critic import critic
from schemas_pwc import Plan, SubQuestion, WorkerAnswer, Verdict
from measure_critic import FAKE_BROKEN, run_single_case

results = []
for case in FAKE_BROKEN:
    name = case["name"]
    print(f"  {name}...", file=sys.stderr)
    fp_00 = run_single_case(case, temperature=0.0, n=10)
    fp_07 = run_single_case(case, temperature=0.7, n=10)
    results.append((name, fp_00, fp_07))
    print(f"    T=0.0: {fp_00}/10, T=0.7: {fp_07}/10", file=sys.stderr)

print(file=sys.stderr)
print("ИТОГ КРИТИКИ:", file=sys.stderr)
for name, fp_00, fp_07 in results:
    print(f"  {name:<35} {fp_00:>5}/10  {fp_07:>5}/10", file=sys.stderr)

# 3. Eval
print(file=sys.stderr)
print("=" * 60, file=sys.stderr)
print("3. EVAL 6x3 (n=3)", file=sys.stderr)
print("=" * 60, file=sys.stderr)

from eval_pwc import CASES, run_case

eval_results = []
for case in CASES:
    qid = case["id"]
    print(f"  {qid}: {case['query'][:60]}...", file=sys.stderr)
    r = run_case(case, n=3)
    eval_results.append(r)
    s = r["single"]; p = r["pwc"]; pv = r["pwc_validator"]
    print(f"    single: {s['pass']}/3  pwc: {p['pass']}/3  pwc+val: {pv['pass']}/3", file=sys.stderr)

# Сохраняем JSON
out_path = Path(__file__).parent / "eval_pwc_results.json"
out_path.write_text(
    json.dumps(eval_results, ensure_ascii=False, indent=2, default=str),
    encoding="utf-8"
)
print(f"\nРезультаты eval: {out_path}", file=sys.stderr)

# Выводим итоговые результаты в stdout
print("=== РЕЗУЛЬТАТЫ ПАРАЛЛЕЛЬНОСТИ ===")
for test in TEST_QUERIES:
    r = par_measure(test["query"], n=3)
    print(f"{test['name']}|{r['avg_seq']:.3f}|{r['avg_par']:.3f}|{r['speedup']:.2f}")

print()
print("=== РЕЗУЛЬТАТЫ КРИТИКИ ===")
for name, fp_00, fp_07 in results:
    print(f"{name}|{fp_00}/10|{fp_07}/10")

print()
print("=== РЕЗУЛЬТАТЫ EVAL ===")
for r in eval_results:
    print(f"{r['id']}|{r['single']['pass']}/3|{r['pwc']['pass']}/3|{r['pwc_validator']['pass']}/3")