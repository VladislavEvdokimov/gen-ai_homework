"""
Замер угодливости Критика: temperature 0.0 против 0.7.

Подготовка 5 заведомо битых наборов ответов, прогон N=10 при T=0.0 и T=0.7.
Подсчёт ложных принятий (verdict.ok=True для битого кейса).

Запуск:
    python measure_critic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from schemas_pwc import Plan, SubQuestion, WorkerAnswer, Verdict

# 5 заведомо битых кейсов
FAKE_BROKEN = [
    # 1. Арифметика без calculate
    {
        "name": "арифметика без calculate",
        "question": "Какова разница между курсом USD и EUR сегодня?",
        "plan": Plan(
            reasoning="Нужно получить курсы USD и EUR и посчитать разницу.",
            subquestions=[
                SubQuestion(id=1, question="Курс USD?", expected_tools=["get_fx_rate"]),
                SubQuestion(id=2, question="Курс EUR?", expected_tools=["get_fx_rate"]),
                SubQuestion(id=3, question="Разница USD-EUR?", expected_tools=["calculate"]),
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Курс USD?",
                answer="USD = 82.5 руб.",
                used_tools=["get_fx_rate"],
                raw_trace=[],
            ),
            2: WorkerAnswer(
                subquestion_id=2,
                question_snippet="Курс EUR?",
                answer="EUR = 89.0 руб.",
                used_tools=["get_fx_rate"],
                raw_trace=[],
            ),
            3: WorkerAnswer(
                subquestion_id=3,
                question_snippet="Разница USD-EUR?",
                answer="Разница = 6.5 руб.",
                used_tools=["get_fx_rate"],  # НЕТ calculate!
                raw_trace=[],
            ),
        },
    },
    # 2. Выдуманное число (нет в данных)
    {
        "name": "выдуманное число",
        "question": "Какой курс USD был 1 января 2022?",
        "plan": Plan(
            reasoning="Получить курс USD на 2022-01-01.",
            subquestions=[
                SubQuestion(id=1, question="Курс USD на 2022-01-01?", expected_tools=["get_fx_rate"]),
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Курс USD на 2022-01-01?",
                answer="USD = 150.0 руб.",  # выдуманное число
                used_tools=["get_fx_rate"],
                raw_trace=[],
            ),
        },
    },
    # 3. Несогласованные данные между подвопросами
    {
        "name": "несогласованные данные",
        "question": "Как изменился курс USD с января 2022 по январь 2023?",
        "plan": Plan(
            reasoning="Получить курсы на две даты и посчитать отношение.",
            subquestions=[
                SubQuestion(id=1, question="Курс USD на 2022-01-01?", expected_tools=["get_fx_rate"]),
                SubQuestion(id=2, question="Курс USD на 2023-01-01?", expected_tools=["get_fx_rate"]),
                SubQuestion(id=3, question="Отношение?", expected_tools=["calculate"]),
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Курс USD на 2022-01-01?",
                answer="USD = 74.0 руб.",
                used_tools=["get_fx_rate"],
                raw_trace=[],
            ),
            2: WorkerAnswer(
                subquestion_id=2,
                question_snippet="Курс USD на 2023-01-01?",
                answer="USD = 82.5 руб.",
                used_tools=["get_fx_rate"],
                raw_trace=[],
            ),
            3: WorkerAnswer(
                subquestion_id=3,
                question_snippet="Отношение?",
                answer="Курс вырос на 150%",  # не согласуется с 74→82.5 (это ~11%, а не 150%)
                used_tools=["calculate"],
                raw_trace=[],
            ),
        },
    },
    # 4. Ответ с ошибкой
    {
        "name": "ответ с ошибкой",
        "question": "Какая инфляция была в январе 2024?",
        "plan": Plan(
            reasoning="Получить ИПЦ за январь 2024.",
            subquestions=[
                SubQuestion(id=1, question="ИПЦ за январь 2024?", expected_tools=["get_inflation"]),
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="ИПЦ за январь 2024?",
                answer="(ошибка: нет данных ИПЦ на 2024-01)",
                used_tools=["get_inflation"],
                raw_trace=[],
            ),
        },
    },
    # 5. План не покрывает вопрос (отсутствует часть вопроса)
    {
        "name": "план не покрывает вопрос",
        "question": "Сравни реальную ключевую ставку сейчас и год назад.",
        "plan": Plan(
            reasoning="Достаточно получить текущую ключевую ставку.",
            subquestions=[
                SubQuestion(id=1, question="Текущая ключевая ставка?", expected_tools=["get_key_rate"]),
            ],
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Текущая ключевая ставка?",
                answer="Ключевая ставка = 21.0%",
                used_tools=["get_key_rate"],
                raw_trace=[],
            ),
        },
    },
]


def run_single_case(
    case: dict,
    temperature: float,
    n: int = 10,
) -> int:
    """Прогнать один кейс N раз, вернуть количество ложных принятий."""
    false_positives = 0
    for i in range(n):
        try:
            verdict = critic(
                question=case["question"],
                plan=case["plan"],
                answers=case["answers"],
                temperature=temperature,
            )
            if verdict.ok:
                false_positives += 1
        except Exception as e:
            print(f"  [i={i}] Ошибка: {e}")
    return false_positives


def main():
    print("=" * 70)
    print("ЗАМЕР УГОДЛИВОСТИ КРИТИКА: T=0.0 vs T=0.7")
    print("=" * 70)
    print()
    print(f"{'Битый кейс':<35} {'T=0.0 ложных':>15} {'T=0.7 ложных':>15}")
    print("-" * 70)

    results = []
    for case in FAKE_BROKEN:
        name = case["name"]
        print(f"\n--- {name} ---")
        
        fp_00 = run_single_case(case, temperature=0.0, n=10)
        fp_07 = run_single_case(case, temperature=0.7, n=10)
        
        results.append((name, fp_00, fp_07))
        print(f"{name:<35} {fp_00:>5}/10 {'':>10} {fp_07:>5}/10")
    
    print()
    print("=" * 70)
    print("ИТОГОВАЯ ТАБЛИЦА:")
    print(f"{'Битый кейс':<35} {'T=0.0 ложных':>15} {'T=0.7 ложных':>15}")
    print("-" * 70)
    for name, fp_00, fp_07 in results:
        print(f"{name:<35} {fp_00:>5}/10 {'':>10} {fp_07:>5}/10")
    
    # Вывод
    print()
    total_00 = sum(fp_00 for _, fp_00, _ in results)
    total_07 = sum(fp_07 for _, _, fp_07 in results)
    print(f"ВСЕГО ложных принятий: T=0.0: {total_00}/50, T=0.7: {total_07}/50")
    
    if total_07 < total_00:
        print("\nВЫВОД: Гипотеза «шум лечит зеркальное соглашение» ПОДТВЕРЖДЕНА:")
        print("  При T=0.7 критик допускает меньше ложных принятий, чем при T=0.0.")
        print("  Шум (температура) помогает критику не повторять ту же логику,")
        print("  что и планировщик, и замечать ошибки.")
    else:
        print("\nВЫВОД: Гипотеза «шум лечит зеркальное соглашение» НЕ ПОДТВЕРЖДЕНА:")
        print("  При T=0.7 критик не показал снижения ложных принятий.")


if __name__ == "__main__":
    main()