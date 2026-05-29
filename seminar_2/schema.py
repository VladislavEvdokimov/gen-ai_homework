"""
Pydantic-схемы для заявок на курсы повышения квалификации (ДПО).
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ───── Списки допустимых значений ─────

CITIES: list[str] = [
    "Москва",
    "Санкт-Петербург",
    "Новосибирск",
    "Екатеринбург",
    "Казань",
    "Красноярск",
    "Нижний Новгород",
    "Челябинск",
    "Самара",
    "Уфа",
    "Ростов-на-Дону",
    "Омск",
]

SPECIALITIES = Literal[
    "учитель",
    "врач",
    "инженер",
    "бухгалтер",
    "менеджер",
    "программист",
    "юрист",
    "экономист",
    "дизайнер",
    "маркетолог",
]

COURSES = Literal[
    "Управление проектами",
    "Цифровой маркетинг",
    "Data Science",
    "Иностранный язык",
    "Педагогика",
    "HR-менеджмент",
    "Финансовый анализ",
    "Психология",
]

current_year = date.today().year

# Минимальный возраст для специальности (с учётом длительности обучения)
SPECIALITY_MIN_AGE: dict[str, int] = {
    "врач": 24,         # мед. вуз 6 лет
    "учитель": 22,      # пед. вуз 4-5 лет
    "инженер": 22,      # техн. вуз 4-5 лет
    "бухгалтер": 22,    # вуз/колледж 3-4 года
    "менеджер": 22,     # вуз 4 года
    "программист": 22,  # вуз / самообучение
    "юрист": 22,        # юр. вуз 4-5 лет
    "экономист": 22,    # вуз 4 года
    "дизайнер": 22,     # вуз / колледж
    "маркетолог": 22,   # вуз 4 года
}


# ───── Вложенная модель Address ─────


class Address(BaseModel):
    city: str
    district: str = Field(min_length=2, max_length=40)

    @field_validator("city")
    @classmethod
    def city_must_be_in_list(cls, v: str) -> str:
        if v not in CITIES:
            raise ValueError(f"Город «{v}» не из утверждённого списка")
        return v


# ───── Основная модель Application ─────


class Application(BaseModel):
    full_name: str
    age: int = Field(ge=22, le=65)
    address: Address
    speciality: SPECIALITIES
    desired_course: COURSES
    years_of_experience: int = Field(ge=0, le=40)
    graduation_year: int = Field(ge=1980, le=2024)

    @model_validator(mode="after")
    def check_age_and_experience(self) -> "Application":
        """
        Межполевые проверки (запускаются после валидации всех полей):
        1. Возраст ≥ минимального для специальности
        2. Стаж ≤ возраст − 18
        """
        # (1) Возраст vs специальность
        min_age = SPECIALITY_MIN_AGE.get(self.speciality, 22)
        if self.age < min_age:
            raise ValueError(
                f"Противоречие: специальность «{self.speciality}», "
                f"возраст {self.age} лет. "
                f"Минимальный возраст для этой специальности: {min_age}."
            )

        # (2) Стаж vs возраст
        max_possible_exp = self.age - 18
        if self.years_of_experience > max_possible_exp:
            raise ValueError(
                f"Противоречие: возраст {self.age}, стаж {self.years_of_experience} лет. "
                f"Максимально возможный стаж: {max_possible_exp} лет "
                f"(с 18 лет)."
            )

        return self

    # Удобный shortcut: app.city работает так же, как app.address.city
    @property
    def city(self) -> str:
        return self.address.city
