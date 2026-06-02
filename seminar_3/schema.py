"""
schema.py — Pydantic-модели для анализа отзывов на мобильное приложение
=======================================================================
Адаптация семинарского пайплайна под Вариант A: отзывы из магазина приложений.

Замены:
  Participant → Review
  concerns   → issues
  Аспекты: производительность, дизайн, поддержка, цена, реклама, надёжность
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ══════════════════════════════════════════════════════════
# Раунд 1 — Information Extraction
# ══════════════════════════════════════════════════════════

# 6 аспектов для Literal (русские названия)
ASPECTS = Literal[
    "производительность",
    "дизайн",
    "поддержка",
    "цена",
    "реклама",
    "надёжность",
]


class Issue(BaseModel):
    """Одна проблема, упомянутая в отзыве."""
    category: ASPECTS
    severity: int = Field(ge=1, le=5, description="Серьёзность от 1 до 5")
    quote: str = Field(description="Точная цитата из отзыва, подтверждающая проблему")

    @field_validator("severity")
    @classmethod
    def severity_in_range(cls, v: int) -> int:
        if v < 1 or v > 5:
            raise ValueError("severity должна быть от 1 до 5")
        return v


class Review(BaseModel):
    """Один отзыв пользователя."""
    id: int = Field(description="Номер отзыва")
    text: str = Field(description="Полный текст отзыва")
    rating: Optional[int] = Field(
        default=None, ge=1, le=5,
        description="Оценка пользователя (1-5), если явно указана",
    )
    review_date: Optional[str] = Field(
        default=None,
        description="Дата отзыва, если указана (строка из текста отзыва)",
    )
    issues: list[Issue] = Field(
        default_factory=list,
        description="Список проблем, которые пользователь упоминает",
    )


class MatchVerdict(BaseModel):
    """Вердикт: совпадает ли тема с эталоном."""
    matched: bool
    matched_index: int = Field(default=-1, description="номер проблемы или -1")
    reason: str = ""


# ══════════════════════════════════════════════════════════
# Раунд 2 — Аспектный анализ
# ══════════════════════════════════════════════════════════

ALL_ASPECTS_LIST: list[str] = [
    "производительность",
    "дизайн",
    "поддержка",
    "цена",
    "реклама",
    "надёжность",
]


class AspectSentiment(BaseModel):
    aspect: ASPECTS
    sentiment: Literal["positive", "negative", "neutral"]
    quote: str = Field(description="Точная цитата-обоснование из отзыва")
    confidence: float = Field(ge=0, le=1)


class ReviewSentiment(BaseModel):
    """Результат аспектного анализа одного отзыва."""
    id: int
    aspects: list[AspectSentiment]


# ══════════════════════════════════════════════════════════
# Раунд 2.5 — Autodiscovery аспектов
# ══════════════════════════════════════════════════════════

class DiscoveredAspect(BaseModel):
    name: str = Field(description="Название темы (англ., snake_case)")
    description: str = Field(min_length=5, description="Описание на русском, 1-2 предложения")


class DiscoveredAspects(BaseModel):
    aspects: list[DiscoveredAspect] = Field(min_length=3, max_length=12)


class DynamicAspect(BaseModel):
    aspect: str
    sentiment: Literal["positive", "negative", "neutral"]
    quote: str
    confidence: float = Field(ge=0, le=1)


class DynamicReview(BaseModel):
    id: int
    aspects: list[DynamicAspect]


# ══════════════════════════════════════════════════════════
# Раунд 3 — Map-Reduce
# ══════════════════════════════════════════════════════════

class ChunkSummary(BaseModel):
    """Резюме группы отзывов (на этапе MAP)."""
    chunk_id: int = Field(description="Номер группы отзывов")
    n_reviews: int = Field(description="Сколько отзывов в группе")
    key_points: list[str] = Field(min_length=1, max_length=6)
    sentiment: Literal["positive", "negative", "mixed"]
    main_aspects: list[str] = Field(
        default_factory=list,
        description="Какие аспекты доминируют в этой группе",
    )


class DiscussionSummary(BaseModel):
    """Итоговое резюме после REDUCE."""
    headline: str = Field(description="Заголовок, отражающий главный вывод")
    key_findings: list[str] = Field(min_length=2, max_length=10)
    action_items: list[str] = Field(min_length=1, max_length=8)


# ══════════════════════════════════════════════════════════
# Раунд 5 — LLM-as-judge
# ══════════════════════════════════════════════════════════

class ActionVerdict(BaseModel):
    action: str
    support: Literal["supported", "weakly_supported", "not_supported"]
    evidence: list[str] = Field(default_factory=list)
    comment: str = Field(description="Почему такой вердикт")


class JudgeReport(BaseModel):
    verdicts: list[ActionVerdict]
    overall_score: float = Field(ge=0, le=1)
    summary: str = Field(description="Общий вывод судьи")


# ══════════════════════════════════════════════════════════
# Раунд 7 — Multi-doc сводка (для «отлично»)
# ══════════════════════════════════════════════════════════

class MultiDocSummary(BaseModel):
    common_themes: list[str] = Field(min_length=1, max_length=8)
    unique_per_group: dict[str, list[str]]
    overall_headline: str
