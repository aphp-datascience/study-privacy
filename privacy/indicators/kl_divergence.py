from scipy.stats import entropy

from privacy.indicators.base import ReliabilityIndicator
from privacy.plots.distribution import DistributionPlot
from privacy.registry import registry
import pandas as pd


@registry.indicators("KLDivergence")
class KLDivergence(ReliabilityIndicator):
    name = "WeekDistribution"

    def __init__(
        self, normalize: bool = False, avoid_last_week: bool = True, **kwargs
    ) -> None:
        """KL Divergence between the distribution of  stays' startweek of the year

        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
        https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
        """
        self.normalize = normalize
        self.avoid_last_week = avoid_last_week

    def compute(self, stays: pd.DataFrame, shifted: bool = False, **kwargs):
        if shifted:
            col_name = "visit_start_date_shifted"
        else:
            col_name = "visit_start_date"

        distribution: pd.Series = (
            stays[col_name]
            .dt.isocalendar()
            .week.value_counts(normalize=self.normalize)
            .sort_index()
            .astype(float)
        )

        distribution
        if self.avoid_last_week:
            return distribution.iloc[:-1]  # Avoid last week of the year
        else:
            return distribution

    def indicators_estimator(self, stays, **kwargs):
        p = self.compute(stays)

        q = self.compute(stays, shifted=True)

        kl = entropy(pk=p, qk=q)

        return {"kl_divergence": kl}

    def get_plot_class(self, **kwargs):
        pc = DistributionPlot(**kwargs)
        return pc
