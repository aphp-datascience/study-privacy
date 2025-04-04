import pandas as pd

from privacy.registry import registry

from .base import BasePseudonymizer


@registry.pseudonymizer("StayPseudonymizer")
class StayPseudonymizer(BasePseudonymizer):
    def __init__(
        self,
        seed: int = 55,
        low_general=-30,
        high_general=30,
        unit_general="D",
        **kwargs,
    ):
        super().__init__(seed, low_general, high_general, unit_general, **kwargs)

    def shift_cohort(self, cohort):
        # Sort values for reproductibility
        cohort.sort_values("person_id", inplace=True)

        # Birth dates (one birth date shift per patient)
        shift_int_birth_date = self.get_random_integer(
            low=self.low_general, high=self.high_general, size=len(cohort)
        )
        cohort["shift_birth_date"] = pd.to_timedelta(
            shift_int_birth_date, unit=self.unit_general
        )

        # Death date (one death date shift per patient)
        shift_int_death_date = self.get_random_integer(
            low=self.low_general, high=self.high_general, size=len(cohort)
        )
        cohort["shift_death_date"] = pd.to_timedelta(
            shift_int_death_date, unit=self.unit_general
        )
        return cohort

    def shift_stays(self, stays: pd.DataFrame):
        # Sort values for reproductibility
        stays.sort_values("visit_occurrence_id", inplace=True)

        # Get random integer for each stay (stay level)
        shift_int = self.get_random_integer(
            low=self.low_general, high=self.high_general, size=len(stays)
        )

        # Shift each stay independently (stay level)
        stays["shift_date"] = pd.to_timedelta(shift_int, unit=self.unit_general)

        return stays

    def pseudonymize(
        self,
        stays: pd.DataFrame,
        cohort: pd.DataFrame,
        date_cols_stay_level=[
            "visit_start_date",
            "visit_end_date",
        ],
        date_cols_patient_level=["birth_date", "death_date"],
        **kwargs,
    ):
        stays = stays.copy()
        # cast to datetime
        self.cast_to_datetime(stays, cols=date_cols_stay_level)
        self.cast_to_datetime(cohort, cols=date_cols_patient_level)

        # Compute visit rank
        self.compute_visit_rank(
            stays,
            sort_values=[
                "visit_start_date",
                "visit_end_date",
            ],
            col_name="visit_number",
        )

        # Get independent shifts for birth and death dates (patient level)
        cohort = self.shift_cohort(cohort)

        # Shift each stay independently (stay level)
        stays = self.shift_stays(stays)

        # Merge cohort info and propagate patient level shifts
        dataset = stays.merge(
            cohort,
            on="person_id",
            how="left",
            validate="many_to_one",
        )

        # Shift date columns
        for col in date_cols_stay_level:
            self.shift_date(dataset, col, "shift_date")

        for col in date_cols_patient_level:
            self.shift_date(dataset, col, "shift_" + col)

        # Compute visit rank after pseudonymization
        self.compute_visit_rank(
            dataset,
            sort_values=["visit_start_date_shifted", "visit_end_date_shifted"],
            col_name="visit_number_shifted",
        )

        return dataset
