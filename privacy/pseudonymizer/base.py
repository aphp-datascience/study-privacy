import numpy as np
import pandas as pd
from numpy.random import PCG64, Generator

from privacy.registry import registry


@registry.pseudonymizer("BasePseudonymizer")
@registry.pseudonymizer(
    "NoPseudonymizer"
)  # Useful to call this class with low_general/high_general = 0 to
# have an output in the same format as pseudonimized data.
class BasePseudonymizer:
    def __init__(
        self,
        seed: int = 55,
        low_general=-30,
        high_general=30,
        unit_general="D",
        **kwargs,
    ):
        self.rng = Generator(PCG64(seed=seed))
        self.set_shifts_params(
            low_general=low_general,
            high_general=high_general,
            unit_general=unit_general,
            **kwargs,
        )

    def get_random_integer(self, low, high, size):
        if (low == 0) and (high == 0):
            shift_int = np.full(shape=size, fill_value=0)
        else:
            shift_int = self.rng.integers(low=low, high=high, size=size)

        return shift_int

    def set_shifts_params(
        self, low_general=-30, high_general=30, unit_general="D", **kwargs
    ):
        self.low_general = low_general
        self.high_general = high_general
        self.unit_general = unit_general

    def shift_cohort(self, cohort):
        # Sort values for reproductibility
        cohort.sort_values("person_id", inplace=True)

        # Get a vector of random integers
        rnd_int_patient_level = self.get_random_integer(
            low=self.low_general, high=self.high_general, size=len(cohort)
        )

        # Set shifts (patient level)
        cohort["shift_date"] = pd.to_timedelta(
            rnd_int_patient_level, unit=self.unit_general
        )
        cohort["shift_birth_date"] = cohort["shift_date"]
        cohort["shift_death_date"] = cohort["shift_date"]
        return cohort

    def get_shifts_params(self):
        return dict(
            general=dict(
                low=pd.to_timedelta(self.low_general, self.unit_general),
                high=pd.to_timedelta(self.high_general, self.unit_general),
            ),
            birth=dict(
                low=pd.to_timedelta(self.low_general, self.unit_general),
                high=pd.to_timedelta(self.high_general, self.unit_general),
            ),
        )

    def shift_date(self, df, col_name, shift_col_name):
        df[col_name + "_shifted"] = df[col_name] + df[shift_col_name]

    @staticmethod
    def cast_to_datetime(
        df,
        cols=[
            "visit_start_date",
            "visit_end_date",
        ],
    ):
        for date_col in cols:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    def compute_visit_rank(
        self,
        stays,
        sort_values=["visit_start_date_shifted", "visit_end_date_shifted"],
        col_name="visit_number_shifted",
    ):
        stays.sort_values(sort_values, inplace=True)
        visit_number = stays.groupby("person_id").cumcount() + 1
        stays[col_name] = visit_number

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
        cohort = cohort.copy()
        self.cast_to_datetime(cohort, cols=date_cols_patient_level)
        self.cast_to_datetime(stays, cols=date_cols_stay_level)

        # Set shift values at patient level
        cohort = self.shift_cohort(cohort)

        # Propagate these shifts to stays
        dataset = stays.merge(
            cohort,
            on="person_id",
            how="left",
            validate="many_to_one",
        )

        # Compute visit rank
        self.compute_visit_rank(
            dataset,
            sort_values=[
                "visit_start_date",
                "visit_end_date",
            ],
            col_name="visit_number",
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
