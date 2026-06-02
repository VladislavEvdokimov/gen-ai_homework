"""
pipeline.py — Полный конвейер анализа отзывов
==============================================
4 базовые техники: IE → Аспекты → Map-Reduce → Judge
+ Prompt Caching (для «отлично»)

Запуск:
    python pipeline.py input/reviews.txt [output_dir]
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from llm_client import get_model, make_client
from prompts import (
    ASPECTS_SYSTEM,
    CHUNK_SYSTEM,
    IE_SYSTEM,
    JUDGE_SYSTEM,
    REDUCE_SYSTEM,
)
from schema import (
    ChunkSummary,
    DiscussionSummary,
    JudgeReport,
    Review,
    ReviewSentiment,
)

client = make_client()
MODEL = get_model()

BATCH_SIZE = 10
COST_PER_1M_TOKENS = 0.15  # DeepSeek V4 approx


# ─── Учёт токенов ────────────────────────────────────

class TokenCounter:
    """Собирает usage со всех запросов для точного подсчёта стоимости."""

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cache_hit_tokens = 0
        self.cache_miss_tokens = 0

    def add(self, usage: Any) -> None:
        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        self.cache_hit_tokens += getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        self.cache_miss_tokens += getattr(usage, "prompt_cache_miss_tokens",
                                          getattr(usage, "prompt_tokens", 0) - self.cache_hit_tokens) or 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost(self) -> float:
        # Cache hit — 10% стоимости, miss — 100%
        hit_cost = self.cache_hit_tokens / 1_000_000 * COST_PER_1M_TOKENS * 0.1
        miss_cost = self.cache_miss_tokens / 1_000_000 * COST_PER_1M_TOKENS
        gen_cost = self.completion_tokens / 1_000_000 * COST_PER_1M_TOKENS
        return round(hit_cost + miss_cost + gen_cost, 6)


counter = TokenCounter()


def tracked_call(**kwargs: Any) -> Any:
    """client.chat.completions.create + учёт токенов."""
    result, completion = client.chat.completions.create(
        **kwargs, with_completion=True,
    )
    if completion and completion.usage:
        counter.add(completion.usage)
    return result


# ─── Парсинг ──────────────────────────────────────────

def parse_reviews(text: str) -> list[dict]:
    """Разобрать reviews.txt в список {'id': int, 'text': str}."""
    reviews = []
    for line in text.strip().split("\n\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("Review "):
            colon = line.find(": ")
            if colon != -1:
                id_str = line[7:colon]
                try:
                    rid = int(id_str.strip())
                    rtext = line[colon + 2:].strip()
                    reviews.append({"id": rid, "text": rtext})
                except ValueError:
                    continue
    return reviews


def split_into_batches(reviews: list[dict], batch_size: int = BATCH_SIZE) -> list[list[dict]]:
    return [reviews[i:i + batch_size] for i in range(0, len(reviews), batch_size)]


def reviews_to_text(batch: list[dict]) -> str:
    parts = []
    for r in batch:
        parts.append(f"Review {r['id']}: {r['text']}")
    return "\n\n".join(parts)


def check_quotes(reviews: list[Review], source_text: str) -> list[tuple[int, str]]:
    t = source_text.lower()
    ghosts: list[tuple[int, str]] = []
    for rev in reviews:
        for issue in rev.issues:
            probe = issue.quote.strip().lower()[:30]
            if probe and probe not in t:
                ghosts.append((rev.id, issue.quote))
    return ghosts


# ─── IE ──────────────────────────────────────────────

def extract_reviews_batch(batch: list[dict]) -> list[Review]:
    batch_text = reviews_to_text(batch)
    return tracked_call(
        model=MODEL,
        response_model=list[Review],
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": IE_SYSTEM},
            {"role": "user", "content": batch_text},
        ],
    )


def extract_all_reviews(reviews_data: list[dict], workers: int = 6) -> list[Review]:
    batches = split_into_batches(reviews_data)
    n = len(batches)
    print(f"  [IE] {len(reviews_data)} otzyvov, {n} batchej po {BATCH_SIZE}, do {workers} parallelno...")

    all_reviews: list[Review] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_reviews_batch, b): i for i, b in enumerate(batches)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                batch_reviews = fut.result()
                all_reviews.extend(batch_reviews)
                print(f"  [IE] batch {i + 1}/{n} gotov ({time.time() - t0:.1f}s)")
            except Exception as e:
                print(f"  [IE] batch {i + 1}/{n} OSHIBKA: {e}")

    all_reviews.sort(key=lambda r: r.id)
    total_issues = sum(len(r.issues) for r in all_reviews)
    print(f"  [IE] vsego: {len(all_reviews)} otzyvov, {total_issues} problem ({time.time() - t0:.1f}s)")
    return all_reviews


# ─── Aspekty ─────────────────────────────────────────

def extract_aspects_batch(batch: list[dict]) -> list[ReviewSentiment]:
    batch_text = reviews_to_text(batch)
    return tracked_call(
        model=MODEL,
        response_model=list[ReviewSentiment],
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": ASPECTS_SYSTEM},
            {"role": "user", "content": batch_text},
        ],
    )


def extract_all_aspects(reviews_data: list[dict], workers: int = 6) -> list[ReviewSentiment]:
    batches = split_into_batches(reviews_data)
    n = len(batches)
    print(f"  [Aspekty] {len(reviews_data)} otzyvov, {n} batchej, do {workers} parallelno...")

    all_aspects: list[ReviewSentiment] = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_aspects_batch, b): i for i, b in enumerate(batches)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                batch_aspects = fut.result()
                all_aspects.extend(batch_aspects)
                print(f"  [Aspekty] batch {i + 1}/{n} gotov ({time.time() - t0:.1f}s)")
            except Exception as e:
                print(f"  [Aspekty] batch {i + 1}/{n} OSHIBKA: {e}")

    all_aspects.sort(key=lambda r: r.id)
    total = sum(len(a.aspects) for a in all_aspects)
    print(f"  [Aspekty] vsego: {len(all_aspects)} otzyvov, {total} ocenok ({time.time() - t0:.1f}s)")
    return all_aspects


def build_heatmap(aspects: list[ReviewSentiment], out_path: str = "output/heatmap.png") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from schema import ALL_ASPECTS_LIST

    group_size = 10
    groups = [aspects[i:i + group_size] for i in range(0, len(aspects), group_size)]
    n_groups = len(groups)

    sent_to_num = {"positive": 1, "negative": -1, "neutral": 0}
    matrix = np.full((n_groups, len(ALL_ASPECTS_LIST)), np.nan)

    for gi, group in enumerate(groups):
        aspect_sums: dict[str, list[float]] = {a: [] for a in ALL_ASPECTS_LIST}
        for rs in group:
            for a in rs.aspects:
                if a.aspect in aspect_sums:
                    aspect_sums[a.aspect].append(sent_to_num[a.sentiment])
        for j, aspect in enumerate(ALL_ASPECTS_LIST):
            vals = aspect_sums[aspect]
            if vals:
                matrix[gi, j] = np.mean(vals)

    plt.figure(figsize=(10, max(5, n_groups * 0.5)))
    sns.heatmap(
        matrix, annot=True, fmt=".1f",
        xticklabels=ALL_ASPECTS_LIST,
        yticklabels=[f"Gruppa {i+1}" for i in range(n_groups)],
        center=0, cmap="RdBu_r",
        cbar_kws={"label": "srednyaya tonalnost"},
    )
    plt.title("Aspektnaya tonalnost po gruppam otzyvov")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  [Aspekty] heatmap: {out_path}")


# ─── Map-Reduce ───────────────────────────────────────

def summarize_chunk(chunk: str, chunk_id: int) -> ChunkSummary:
    return tracked_call(
        model=MODEL,
        response_model=ChunkSummary,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": CHUNK_SYSTEM},
            {"role": "user", "content": chunk},
        ],
    )


def reduce_summaries(summaries: list[ChunkSummary]) -> DiscussionSummary:
    joined = "\n\n".join(
        f"## Gruppa {s.chunk_id} ({s.sentiment}, {s.n_reviews} otzyvov)\n"
        + "\n".join(f"- {p}" for p in s.key_points)
        + ("\nAspekty: " + ", ".join(s.main_aspects) if s.main_aspects else "")
        for s in summaries
    )
    return tracked_call(
        model=MODEL,
        response_model=DiscussionSummary,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": REDUCE_SYSTEM},
            {"role": "user", "content": joined},
        ],
    )


def map_reduce(reviews_data: list[dict], workers: int = 6) -> DiscussionSummary:
    batches = split_into_batches(reviews_data, batch_size=BATCH_SIZE)
    n = len(batches)
    print(f"  [MR] MAP: {n} grupp, do {workers} parallelno...")

    t0 = time.time()
    chunk_summaries: list[ChunkSummary | None] = [None] * n

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for i, batch in enumerate(batches):
            batch_text = reviews_to_text(batch)
            futures[pool.submit(summarize_chunk, batch_text, i + 1)] = i

        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                chunk_summaries[i] = fut.result()
                done += 1
                print(f"  [MR] {done}/{n} gotovo ({time.time() - t0:.1f}s)")
            except Exception as e:
                print(f"  [MR] gruppa {i+1}/{n} OSHIBKA: {e}")

    valid = [s for s in chunk_summaries if s is not None]
    print(f"  [MR] MAP {time.time() - t0:.1f}s -> REDUCE ({len(valid)} grupp)...")
    result = reduce_summaries(valid)
    print(f"  [MR] vsego {time.time() - t0:.1f}s")
    return result


# ─── Judge ────────────────────────────────────────────

def build_evidence_packet(reviews: list[Review], summary: DiscussionSummary) -> str:
    parts = ["## Rekomendacii (kotorye ocenivaem)"]
    for i, a in enumerate(summary.action_items, 1):
        parts.append(f"  {i}. {a}")

    parts.append("\n## Problemy iz otzyvov (ishodnye dannye)")
    for rev in reviews:
        for issue in rev.issues:
            parts.append(
                f"  - [Otzyv {rev.id}/{issue.category}, sev={issue.severity}] "
                f"'{issue.quote}'"
            )
    return "\n".join(parts)


def judge_reviews(reviews: list[Review], summary: DiscussionSummary) -> JudgeReport:
    evidence = build_evidence_packet(reviews, summary)
    return tracked_call(
        model=MODEL,
        response_model=JudgeReport,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": evidence},
        ],
    )


# ─── Prompt Caching ───────────────────────────────────

def run_caching_demo(reviews_data: list[dict]) -> dict:
    from schema import ReviewSentiment as RS

    batch = reviews_data[:5]
    batch_text = reviews_to_text(batch)

    results = []
    labels = []

    for label, prompt in [
        ("Holodnyj start", ASPECTS_SYSTEM),
        ("Povtor (kesh)", ASPECTS_SYSTEM),
    ]:
        t0 = time.time()
        _result, completion = client.chat.completions.create(
            model=MODEL,
            response_model=list[RS],
            max_retries=2,
            temperature=0.0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": batch_text},
            ],
            with_completion=True,
        )
        usage = completion.usage
        cache_hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        cache_miss = getattr(usage, "prompt_cache_miss_tokens", usage.prompt_tokens - cache_hit) or 0
        results.append({
            "time": round(time.time() - t0, 1),
            "prompt_tokens": usage.prompt_tokens,
            "cache_hit": cache_hit,
            "cache_miss": cache_miss,
        })
        labels.append(label)

    return {"labels": labels, "results": results}


# ─── Analyze ──────────────────────────────────────────

def analyze(input_path: str, out_dir: str = "output") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipeline_start = time.time()

    print("=" * 50)
    print("ZAPUSK KONVEJERA ANALIZA OTZYVOV")
    print("=" * 50)
    raw_text = Path(input_path).read_text(encoding="utf-8")
    reviews_data = parse_reviews(raw_text)
    print(f"Prochitano: {len(reviews_data)} otzyvov iz {input_path}\n")

    # Akt 1: IE
    print("-" * 40)
    print("AKT 1: Information Extraction")
    print("-" * 40)
    t0 = time.time()
    reviews = extract_all_reviews(reviews_data)
    ie_time = time.time() - t0

    reviews_data_out = [r.model_dump() for r in reviews]
    (out / "reviews.json").write_text(
        json.dumps(reviews_data_out, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    n_valid = len(reviews)
    n_errors = len(reviews_data) - n_valid
    total_issues = sum(len(r.issues) for r in reviews)
    print(f"✓ IE: {n_valid} validnyh, {n_errors} oshibok, {total_issues} problem\n")

    # Ghost-cytat
    print("-" * 40)
    print("PROVERKA: ghost-cytat (check_quotes)")
    print("-" * 40)
    ghosts = check_quotes(reviews, raw_text)
    n_ghosts = len(ghosts)
    total_quotes = sum(len(r.issues) for r in reviews)
    ghost_pct = n_ghosts / total_quotes * 100 if total_quotes else 0
    print(f"  Vsego citat: {total_quotes}")
    print(f"  Ghost-citat: {n_ghosts} ({ghost_pct:.1f}%)")
    if ghosts:
        for rid, q in ghosts[:5]:
            print(f"    X Otzyv {rid}: '{q[:80]}'")
        if len(ghosts) > 5:
            print(f"    ... i eshe {len(ghosts)-5}")
    if ghost_pct <= 10:
        print("  ✅ ghost-citat <= 10% - OK\n")
    else:
        print(f"  ⚠ ghost-citat > 10% - nuzhno uluchshit prompt\n")

    # Akt 2: Aspekty
    print("-" * 40)
    print("AKT 2: Aspektnyj analiz")
    print("-" * 40)
    t0 = time.time()
    aspects = extract_all_aspects(reviews_data)
    aspects_time = time.time() - t0

    aspects_data = [a.model_dump() for a in aspects]
    (out / "aspects.json").write_text(
        json.dumps(aspects_data, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    build_heatmap(aspects, out_path=str(out / "heatmap.png"))
    total_aspect_evals = sum(len(a.aspects) for a in aspects)
    print(f"✓ Aspekty: {total_aspect_evals} ocenok po {len(aspects)} otzyvam\n")

    # Akt 3: Map-Reduce
    print("-" * 40)
    print("AKT 3: Map-Reduce reziume")
    print("-" * 40)
    t0 = time.time()
    summary = map_reduce(reviews_data)
    mr_time = time.time() - t0

    (out / "summary.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8",
    )
    print(f"\n  Zagolovok: {summary.headline}")
    print(f"  Vyvodov: {len(summary.key_findings)}")
    print(f"  Rekomendacij: {len(summary.action_items)}\n")

    # Akt 5: Judge
    print("-" * 40)
    print("AKT 5: LLM-as-judge")
    print("-" * 40)
    t0 = time.time()
    report = judge_reviews(reviews, summary)
    judge_time = time.time() - t0

    (out / "judge_report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8",
    )

    counts = {"supported": 0, "weakly_supported": 0, "not_supported": 0}
    for v in report.verdicts:
        counts[v.support] += 1
    print(f"  supported:        {counts['supported']}")
    print(f"  weakly_supported: {counts['weakly_supported']}")
    print(f"  not_supported:    {counts['not_supported']}")
    print(f"  overall_score:    {report.overall_score:.2f}")
    print(f"  {report.summary}")

    if report.overall_score < 0.7:
        print("\n⚠ overall_score < 0.7. Rekomenduetsya perepisat REDUCE prompt i zapustit zanovo.")

    # Caching
    print("-" * 40)
    print("DOPOLNITELNO: Prompt Caching (dlya otlichno)")
    print("-" * 40)
    t0 = time.time()
    cache_result = run_caching_demo(reviews_data)
    cache_time = time.time() - t0

    for label, res in zip(cache_result["labels"], cache_result["results"]):
        total = res["prompt_tokens"]
        hit, miss = res["cache_hit"], res["cache_miss"]
        pct = hit / total * 100 if total else 0
        print(f"  {label:<20} t={res['time']:>5.1f}s  "
              f"tokenov={total:>5}  popadanij={hit:>5} ({pct:>3.0f}%)  "
              f"promahov={miss:>5}")

    second = cache_result["results"][1]
    if second["cache_hit"] > 0:
        savings = second["cache_hit"] / 1_000_000 * COST_PER_1M_TOKENS * 0.9
        print(f"\n  💰 Ekonomiya ot kesha: ${savings:.6f} "
              f"(na {second['cache_hit']} zakeshirovannyh tokenah)")

    (out / "cache_report.json").write_text(
        json.dumps(cache_result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print()

    # Metriki
    print("-" * 40)
    print("ITOGOVYE METRIKI")
    print("-" * 40)
    pipeline_time = time.time() - pipeline_start

    print(f"  Vsego zaprosov (pribl.): {len(reviews_data)//BATCH_SIZE * 4 + 1}")
    print(f"  Prompt tokens: {counter.prompt_tokens}")
    print(f"  Completion tokens: {counter.completion_tokens}")
    print(f"  Cache hit tokens: {counter.cache_hit_tokens}")
    print(f"  Cache miss tokens: {counter.cache_miss_tokens}")
    print(f"  Itogovaya stoimost: ${counter.cost:.6f}")

    metrics = {
        "n_reviews": len(reviews_data),
        "n_valid_reviews": n_valid,
        "n_errors": n_errors,
        "n_issues": total_issues,
        "n_ghost_quotes": n_ghosts,
        "ghost_pct": round(ghost_pct, 1),
        "overall_score": report.overall_score,
        "time_seconds": {
            "ie": round(ie_time, 1),
            "aspects": round(aspects_time, 1),
            "map_reduce": round(mr_time, 1),
            "judge": round(judge_time, 1),
            "caching": round(cache_time, 1),
            "total": round(pipeline_time, 1),
        },
        "token_usage": {
            "prompt_tokens": counter.prompt_tokens,
            "completion_tokens": counter.completion_tokens,
            "cache_hit_tokens": counter.cache_hit_tokens,
            "cache_miss_tokens": counter.cache_miss_tokens,
        },
        "estimated_cost_usd": counter.cost,
        "judge_counts": counts,
        "caching": cache_result,
    }

    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print(f"\n{'=' * 50}")
    print("KONVEJER ZAVERSHEN")
    print(f"{'=' * 50}")
    print(f"\n{summary.headline}")
    print(f"\nKluchevye vyvody:")
    for kf in summary.key_findings:
        print(f"  * {kf}")
    print(f"\nRekomendacii:")
    for ai in summary.action_items:
        print(f"  -> {ai}")
    print(f"\nOcenka sudi: {report.overall_score:.2f}")
    print(f"Ghost-citat: {n_ghosts}/{total_quotes} ({ghost_pct:.1f}%)")
    print(f"Vremya progona: {pipeline_time:.1f}s")
    print(f"Itogovaya stoimost: ${counter.cost:.6f}")
    print(f"\nVse artefakty: {out}/")


def main() -> None:
    if len(sys.argv) < 2:
        print("Ispolzovanie: python pipeline.py <input.txt> [out_dir]")
        print("  Primer: python pipeline.py input/reviews.txt output")
        sys.exit(1)
    analyze(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "output")


if __name__ == "__main__":
    main()
