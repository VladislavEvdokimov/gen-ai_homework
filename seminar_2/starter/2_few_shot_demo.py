"""
Раунд 1.5 — Few-shot из seed_examples.json
==========================================
После раунда 1 у нас включён JSON mode, и markdown-обёртки исчезли. Но мы
ещё ничего не сказали модели про *структуру*: какие именно поля, какие
города, какие профессии. Поэтому модель импровизирует.

Дешёвый способ дать ей образец — few-shot: показываем 2-3 примера
правильных ответов в самом промпте. Это слайд про in-context learning из
лекции 2.

Запуск:
  python few_shot_demo.py
"""

import json
import time

from llm_client import get_model, make_raw_client

client = make_raw_client()
MODEL = get_model()

with open("seed_examples.json", encoding="utf-8") as f:
    SEEDS = json.load(f)


def build_few_shot_prompt(n_shots: int = 2) -> str:
    """Собрать SYSTEM_PROMPT с n примерами «правильного» ответа."""
    examples = "\n\n".join(
        f"Пример {i + 1}:\n{json.dumps(s, ensure_ascii=False, indent=2)}"
        for i, s in enumerate(SEEDS[:n_shots])
    )
    return (
        "Ты — генератор синтетических покупательских персон для российского "
        "e-commerce. Верни ОДИН JSON-объект с теми же полями, что в "
        "примерах ниже. Никаких пояснений, никакого markdown.\n\n"
        f"{examples}\n\n"
        "Сгенерируй НОВУЮ персону в том же формате."
    )


def run(n_shots: int, n_runs: int = 5) -> dict:
    sys_prompt = build_few_shot_prompt(n_shots)
    parsed, has_all_fields, age_int = 0, 0, 0
    required = {
        "name",
        "age",
        "city",
        "income_rub",
        "occupation",
        "shopping_frequency",
        "preferred_category",
    }

    print(f"━━━ few-shot: {n_shots} примеров, {n_runs} запросов ━━━")
    for i in range(n_runs):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": "Создай одну персону."},
            ],
            # JSON mode уже включён — это раунд 1
            response_format={"type": "json_object"},
            temperature=0.9,
        )
        text = (resp.choices[0].message.content or "").strip()
        try:
            data = json.loads(text)
            parsed += 1
            has_all = required.issubset(data.keys())
            has_all_fields += int(has_all)
            age_int += int(isinstance(data.get("age"), int))
            mark_fields = "all" if has_all else f"miss={required - set(data.keys())}"
            mark_age = (
                "int"
                if isinstance(data.get("age"), int)
                else f"{type(data.get('age')).__name__}"
            )
            print(f"  [{i + 1}/{n_runs}] ✓ fields={mark_fields} age={mark_age}")
        except json.JSONDecodeError as e:
            print(f"  [{i + 1}/{n_runs}] ✗ {e}")
        time.sleep(0.3)
    return {"parsed": parsed, "has_all": has_all_fields, "age_int": age_int}


def main():
    print(f"Модель: {MODEL}\n")
    print("Сравним: 0 примеров (просто инструкция) vs 2 примера (few-shot)\n")

    no_shots = run(n_shots=0, n_runs=5)
    print()
    with_shots = run(n_shots=2, n_runs=5)

    print("\n━━━ Сводка ━━━")
    print(f"{'shots':<8} {'JSON':>6} {'все поля':>10} {'age=int':>9}")
    print(
        f"{'0':<8} {no_shots['parsed']}/5    {no_shots['has_all']}/5        {no_shots['age_int']}/5"
    )
    print(
        f"{'2':<8} {with_shots['parsed']}/5    {with_shots['has_all']}/5        {with_shots['age_int']}/5"
    )
    print("\nВывод: few-shot сильно повышает шанс получить все поля и нужные типы,")
    print("но НЕ заменяет валидацию. В раунде 2 ставим Pydantic поверх.")


if __name__ == "__main__":
    main()
