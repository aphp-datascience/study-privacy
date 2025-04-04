from typing import Optional

import numpy as np
import pandas as pd

from privacy.indicators.base import ReliabilityIndicator
from privacy.plots.box_plot import BoxPlot
from privacy.registry import registry


@registry.indicators("Age")
class Age(ReliabilityIndicator):
    name = "Age"

    def __init__(
        self,
        floor: bool = False,
        filter_first_visit: bool = False,
        return_variance: bool = True,
        unit: str = "Y",
        **kwargs,
    ):
        # Parameters
        self.floor = floor
        self.filter_first_visit = filter_first_visit
        self.return_variance = return_variance
        self.unit = unit
        # column names
        self._set_cols()

    def _set_cols(
        self,
        date_col: str = "visit_start_date",
        birth_col: str = "birth_date",
        visit_number_col: str = "visit_number",
    ):
        self.date_col = date_col
        self.birth_col = birth_col
        self.visit_number_col = visit_number_col

    @staticmethod
    def _compute_age(
        df: pd.DataFrame,
        date_col: str = "visit_start_date",
        birth_col: str = "birth_date",
        floor: bool = True,
        unit: str = "Y",
    ):
        # Compute age
        age = (df[date_col] - df[birth_col]) / np.timedelta64(1, unit)

        if floor:
            age = (np.floor(age)).astype(pd.Int64Dtype())

        return age

    def compute(self, stays: pd.DataFrame, shifted: bool = False, **kwargs):
        if shifted:
            self._set_cols(
                date_col="visit_start_date_shifted",
                birth_col="birth_date_shifted",
                visit_number_col="visit_number_shifted",
            )
        if self.filter_first_visit:
            stays = stays.query(f"{self.visit_number_col} == 1")

        age = self._compute_age(
            stays,
            date_col=self.date_col,
            birth_col=self.birth_col,
            floor=self.floor,
            unit=self.unit,
        )

        # Reset cols
        self._set_cols()

        return age

    def mean_and_variance(
        self,
        age: pd.Series,
    ):
        mean = age.mean()

        variance = age.std() ** 2

        return mean, variance

    def indicators_estimator(self, stays: pd.DataFrame, **kwargs):
        """Compute mean and variance indicators for age at the given stays.

        Parameters
        ----------
        stays : pd.DataFrame
            stays to compute age indicators

        Returns
        -------
            Dict[`mean_age_indicator`, `variance_age_indicator`]
        """
        # Mean and variance of Real Age
        age0 = self.compute(stays)
        mean0, variance0 = self.mean_and_variance(age0)

        # Mean and variance of shifted age
        age = self.compute(stays, shifted=True)
        mean_shifted, variance_shifted = self.mean_and_variance(age)

        # Indicators
        s1 = np.abs(mean_shifted - mean0) / mean0
        s2 = np.abs(variance_shifted - variance0) / variance0
        results = {"mean_age_indicator": s1}
        if self.return_variance:
            results.update({"variance_age_indicator": s2})
        return results

    def get_plot_class(self, **kwargs):
        pc = BoxPlot(**kwargs)
        return pc
