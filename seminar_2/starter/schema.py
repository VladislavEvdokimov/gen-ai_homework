"""
Pydantic-схема для персоны.
==========================
Сейчас здесь только комментарии.
(раскомментируй и допиши на семинаре):
"""

# ───── Раунды 2–4: плоская Persona ─────
# from typing import Literal
# from pydantic import BaseModel, Field, field_validator
#
# CITIES = {
#     "Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург",
#     "Казань", "Нижний Новгород", "Самара", "Краснодар",
# }
#
#
# class Persona(BaseModel):
#     name: str
#     age: int = Field(ge=18, le=75)
#     city: str
#     income_rub: int = Field(ge=30_000, le=500_000)
#     occupation: Literal[
#         "студент", "инженер", "менеджер", "учитель",
#         "врач", "предприниматель", "пенсионер", "IT-специалист",
#     ]
#     shopping_frequency: Literal["редко", "иногда", "часто"]
#     preferred_category: Literal[
#         "электроника", "одежда", "продукты", "книги",
#         "товары для дома", "косметика", "спорт",
#     ]
#
#     @field_validator("city")
#     @classmethod
#     def city_must_be_in_list(cls, v: str) -> str:
#         if v not in CITIES:
#             raise ValueError(f"Город «{v}» не из утверждённого списка")
#         return v


# ───── Раунд 4.5: вложенная Address ─────
# class Address(BaseModel):
#     city: str
#     district: str = Field(min_length=2, max_length=40)
#
#     @field_validator("city")
#     @classmethod
#     def city_must_be_in_list(cls, v: str) -> str:
#         if v not in CITIES:
#             raise ValueError(f"Город «{v}» не из утверждённого списка")
#         return v
#
#
# class Persona(BaseModel):
#     name: str
#     age: int = Field(ge=18, le=75)
#     address: Address       # ← было city: str, стало вложенный объект
#     income_rub: int = Field(ge=30_000, le=500_000)
#     occupation: Literal[
#         "студент", "инженер", "менеджер", "учитель",
#         "врач", "предприниматель", "пенсионер", "IT-специалист",
#     ]
#     shopping_frequency: Literal["редко", "иногда", "часто"]
#     preferred_category: Literal[
#         "электроника", "одежда", "продукты", "книги",
#         "товары для дома", "косметика", "спорт",
#     ]
#
#     # Удобный shortcut для check_personas.py и 5_analysis.py —
#     # persona.city работает что для плоской, что для вложенной схемы.
#     @property
#     def city(self) -> str:
#         return self.address.city
