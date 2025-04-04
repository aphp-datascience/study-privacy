import pandas as pd

from privacy.registry import registry

from .base import BasePseudonymizer


@registry.pseudonymizer("BirthPseudonymizer")
class BirthPseudonymizer(BasePseudonymizer):
    def __init__(
        self,
        seed: int = 55,
        low_general=-30,
        high_general=30,
        unit_general="D",
        low_birth_date=-5,
        high_birth_date=5,
        unit_birth_date="D",
        **kwargs,
    ):
        super().__init__(seed, low_general, high_general, unit_general, **kwargs)
        self.set_shifts_params(
            low_general=low_general,
            high_general=high_general,
            unit_general=unit_general,
            low_birth_date=low_birth_date,
            high_birth_date=high_birth_date,
            unit_birth_date=unit_birth_date,
            **kwargs,
        )

    def set_shifts_params(
        self,
        low_general=-30,
        high_general=30,
        unit_general="D",
        low_birth_date=-5,
        high_birth_date=5,
        unit_birth_date="D",
        **kwargs,
    ):
        self.low_general = low_general
        self.high_general = high_general
        self.unit_general = unit_general
        self.low_birth_date = low_birth_date
        self.high_birth_date = high_birth_date
        self.unit_birth_date = unit_birth_date

    def shift_cohort(self, cohort):
        # Sort values for reproductibility
        cohort.sort_values("person_id", inplace=True)

        # Non birth dates
        shift_int = self.get_random_integer(
            low=self.low_general, high=self.high_general, size=len(cohort)
        )
        cohort["shift_date"] = pd.to_timedelta(shift_int, unit=self.unit_general)
        cohort["shift_death_date"] = cohort["shift_date"]

        # Birth dates
        shift_int_birth_date = shift_int + self.get_random_integer(
            low=self.low_birth_date, high=self.high_birth_date, size=len(cohort)
        )

        cohort["shift_birth_date"] = pd.to_timedelta(
            shift_int_birth_date, unit=self.unit_birth_date
        )
        return cohort

    def get_shifts_params(self):
        return dict(
            general=dict(
                low=pd.to_timedelta(self.low_general, self.unit_general),
                high=pd.to_timedelta(self.high_general, self.unit_general),
            ),
            birth=dict(
                low=pd.to_timedelta(self.low_general, self.unit_general)
                + pd.to_timedelta(self.low_birth_date, self.unit_birth_date),
                high=pd.to_timedelta(self.high_general, self.unit_general)
                + pd.to_timedelta(self.high_birth_date, self.unit_birth_date),
            ),
        )
