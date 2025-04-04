import sys

import numpy as np
import pandas as pd
from loguru import logger

from privacy.indicators.base import ReliabilityIndicator
from privacy.plots.proportions import ProportionsPlot
from privacy.registry import registry

logger.remove()
logger.add(sys.stderr, level="INFO")


@registry.indicators("Readmission")
class Readmission(ReliabilityIndicator):
    name = "Readmission"

    def __init__(
        self,
        readmission_max_delay_value: str,
        criteria_col: str = "first_stay",
        **kwargs,
    ) -> None:
        self.criteria_col = criteria_col
        self.readmission_timedelta = pd.Timedelta(readmission_max_delay_value)

    def process(self, stays: pd.DataFrame, col_suffix: str = ""):
        visit_number_col = "visit_number" + col_suffix
        visit_end_date_col = "visit_end_date" + col_suffix
        visit_start_date_col = "visit_start_date" + col_suffix

        if self.criteria_col == "first_stay":
            stays["first_stay"] = stays[visit_number_col] == 1

        # First stay and with end date;
        criteria_stay = stays.query(
            f"{self.criteria_col} & {visit_end_date_col}.notna()"
        )[
            [
                "person_id",
                "visit_occurrence_id",
                visit_start_date_col,
                visit_end_date_col,
            ]
        ].sort_values(visit_end_date_col, inplace=False)

        # second stay
        all_stays = stays[
            [
                "person_id",
                "visit_occurrence_id",
                visit_start_date_col,
            ]
        ].sort_values(visit_start_date_col, inplace=False)

        criteria_stay_and_next = pd.merge_asof(
            criteria_stay,
            all_stays,
            by="person_id",
            left_on=visit_end_date_col,
            right_on=visit_start_date_col,
            tolerance=self.readmission_timedelta,
            direction="forward",
            allow_exact_matches=False,
            suffixes=("", "_2"),
        )

        criteria_stay_and_next["readmitted"] = (
            criteria_stay_and_next.visit_occurrence_id_2.notna()
        )

        return criteria_stay_and_next

    def compute(self, stays, shifted: bool = False, **kwargs):
        if shifted:
            col_suffix = "_shifted"
        else:
            col_suffix = ""
        first_and_second_stays = self.process(stays=stays, col_suffix=col_suffix)
        readmission_rate = first_and_second_stays.readmitted.value_counts(
            normalize=True
        ).loc[True]

        return readmission_rate

    def indicators_estimator(self, stays, **kwargs):
        readmission_rate_0 = self.compute(stays)
        readmission_rate_p = self.compute(stays, shifted=True)

        s3 = np.abs(readmission_rate_p - readmission_rate_0) / readmission_rate_0
        logger.debug(f"readmission_rate_0 {readmission_rate_0}")
        logger.debug(f"readmission_rate_p {readmission_rate_p}")

        return {
            "readmission_indicator": s3,
        }

    def get_plot_class(self, **kwargs):
        pc = ProportionsPlot(**kwargs)
        return pc
