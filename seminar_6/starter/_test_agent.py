"""Краткий тест: работает ли агент?"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_s5 import run_agent
from orchestrator import run_pwc

# Тест 1: простой запрос к agent_s5
print("=== Тест agent_s5 ===")
try:
    r = run_agent("Какой курс USD?", max_iter=3, verbose=False)
    ans = r.get("answer")
    err = r.get("error")
    steps = r.get("steps")
    print(f"answer={str(ans)[:80] if ans else None}")
    print(f"error={err}")
    print(f"steps={steps}")
except Exception as e:
    print(f"EXCEPTION: {type(e).__name__}: {e}")

# Тест 2: простой запрос через PWC
print("\n=== Тест run_pwc ===")
try:
    r = run_pwc("Какой курс USD?", max_iter=2, verbose=False, use_validator=True)
    ans = r.get("answer")
    err = r.get("error")
    print(f"answer={str(ans)[:80] if ans else None}")
    print(f"error={err}")
except Exception as e:
    print(f"EXCEPTION: {type(e).__name__}: {e}")