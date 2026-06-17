from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
from enum import Enum

# ─── Сентимент ───
class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

# ─── Персона ───
class Persona(BaseModel):
    id: int
    name: str
    age: int = Field(ge=18, le=90)
    gender: Literal["male", "female", "non-binary"]
    occupation: str
    income_level: Literal["low", "middle", "high"]
    residence: Literal["city", "suburb", "rural"]
    car_experience: str
    attitude_to_hybrid: Literal["positive", "neutral", "negative", "undecided"]
    personality_traits: str  # краткое описание характера

    @field_validator("age")
    @classmethod
    def age_in_range(cls, v: int) -> int:
        if v < 18 or v > 90:
            raise ValueError(f"Age must be between 18 and 90, got {v}")
        return v

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Name must not be empty")
        return v

# ─── Реплика в дискуссии ───
class Utterance(BaseModel):
    persona_id: int
    persona_name: str
    text: str
    round: int = Field(ge=1, le=20)

# ─── Транскрипт ───
class Transcript(BaseModel):
    topic: str
    utterances: List[Utterance]

# ─── Извлечённая сущность ───
class Entity(BaseModel):
    persona_id: int
    persona_name: str
    product_aspect: str = Field(description="Аспект продукта: цена, надёжность, экология, дизайн, расход и т.д.")
    opinion: str = Field(description="Мнение персоны об этом аспекте")
    sentiment: Sentiment
    quote: str = Field(description="Прямая цитата из транскрипта, подтверждающая мнение")

    @field_validator("sentiment")
    @classmethod
    def sentiment_valid(cls, v: Sentiment) -> Sentiment:
        return v

# ─── Аспектная группа ───
class AspectGroup(BaseModel):
    aspect: str
    opinions: List[Entity]

# ─── Результат аспектного анализа ───
class AspectAnalysis(BaseModel):
    groups: List[AspectGroup]

# ─── Map-Reduce: сводка по аспекту ───
class AspectSummary(BaseModel):
    aspect: str
    summary: str = Field(description="Итоговое резюме мнений по аспекту на основе всех реплик")
    consensus: Literal["consensus", "mixed", "disagreement"]
    key_concerns: List[str] = Field(description="Ключевые проблемы/тезисы, выделенные участниками")

# ─── Итоговый отчёт ───
class FocusGroupReport(BaseModel):
    topic: str
    summaries: List[AspectSummary]
    overall_sentiment: Sentiment
    recommendations: List[str] = Field(description="Рекомендации производителю на основе дискуссии")

# ─── Результат LLM-as-judge ───
class JudgeEvaluation(BaseModel):
    persona_consistency_score: int = Field(ge=1, le=5, description="Насколько каждая персона придерживалась своей позиции")
    hallucination_count: int = Field(ge=0, description="Количество ghost-цитат / выдуманных фактов")
    hallucination_examples: List[str] = Field(default_factory=list, description="Примеры галлюцинаций")
    transcript_grounding_score: int = Field(ge=1, le=5, description="Насколько выводы привязаны к тексту транскрипта")
    overall_quality_score: int = Field(ge=1, le=5, description="Общее качество симуляции")
    comments: str = Field(description="Замечания и рекомендации по улучшению")

    @field_validator("persona_consistency_score")
    @classmethod
    def score_in_range_1_5(cls, v: int) -> int:
        if v < 1 or v > 5:
            raise ValueError(f"Score must be between 1 and 5, got {v}")
        return v

    @field_validator("transcript_grounding_score")
    @classmethod
    def grounding_in_range_1_5(cls, v: int) -> int:
        if v < 1 or v > 5:
            raise ValueError(f"Score must be between 1 and 5, got {v}")
        return v

    @field_validator("overall_quality_score")
    @classmethod
    def quality_in_range_1_5(cls, v: int) -> int:
        if v < 1 or v > 5:
            raise ValueError(f"Score must be between 1 and 5, got {v}")
        return v


# ─── Eval: один тестовый случай ───
class EvalCase(BaseModel):
    case_id: int
    persona_seed: int
    topic_variant: str
    transcript: Optional[Transcript] = None
    entities: Optional[List[Entity]] = None
    report: Optional[FocusGroupReport] = None
    judge: Optional[JudgeEvaluation] = None
    cost: float = 0.0
    steps: int = 0
    tokens: int = 0
    passed: bool = False
    error: Optional[str] = None