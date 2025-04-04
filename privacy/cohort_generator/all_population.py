from typing import List, Optional

from privacy.registry import registry

from .base import BaseCohortGenerator


@registry.cohort_generator("AllPopulation")
class AllPopulation(BaseCohortGenerator):
    def __init__(
        self,
        db: str,
        start_date: str,
        end_date: str,
        visit_type: List[str] = ["hospitalisés"],
        age_min_at_stay: Optional[int] = 0,
    ) -> None:
        super().__init__(db, start_date, end_date, visit_type, age_min_at_stay)
