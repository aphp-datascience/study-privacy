from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from statsmodels.duration import hazard_regression

from privacy.indicators.base import ReliabilityIndicator
from privacy.plots.kaplan_meier import km_plot
from privacy.registry import registry


@registry.indicators("SurvivalAnalysis")
class SurvivalAnalysis(ReliabilityIndicator):
    name = "Survival"

    def __init__(
        self,
        date_end_study: str = "2020-02-01",
        exogeneous_variables=[
            "age_at_diagnostic",
            "sex",
            "anticancer_therapy",
            "surgery",
        ],
    ) -> None:
        self.date_end_study = pd.to_datetime(date_end_study)
        self.exogeneous_variables = exogeneous_variables
        pass

    def preprocess(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        assert "first_diagnostic_code" in dataset.columns
        # Avoid real diagnostic codes after death
        filtered_dataset = dataset.query(
            "(first_diagnostic_code < death_date) | (death_date.isna()) "
        ).copy()

        # Shift diagnositc date in consistent way of the rest of event dates
        if "first_diagnostic_code_shifted" not in filtered_dataset.columns:
            filtered_dataset["first_diagnostic_code_shifted"] = (
                filtered_dataset["first_diagnostic_code"]
                + filtered_dataset["shift_date"]
            )

        # Encode sex
        filtered_dataset["sex"] = (filtered_dataset.gender_source_value == "m").astype(
            int
        )
        return filtered_dataset

    def get_data_to_fit(
        self,
        dataset: pd.DataFrame,
        visit_number_col_name: str = "visit_number",
        birth_date_col_name: str = "birth_date",
        death_date_col_name: str = "death_date",
        first_diagnostic_code_col_name: str = "first_diagnostic_code",
    ) -> pd.DataFrame:
        # Get first stay
        dataset_first = dataset.loc[dataset[visit_number_col_name] == 1].copy()

        # Make death status variable
        dataset_first["death"] = (
            dataset_first.death_date < self.date_end_study
        )  # if the patient dies after end_study, so he is not considered death

        # Compute age at diagnostic
        dataset_first["age_at_diagnostic"] = (
            dataset_first[first_diagnostic_code_col_name]
            - dataset_first[birth_date_col_name]
        ).dt.days / 365

        # Encode treatment (of the first stay)
        dummies_treatment = pd.get_dummies(dataset_first.treatment)
        dataset_first = pd.concat([dataset_first, dummies_treatment], axis=1)
        treatment_names = list(dummies_treatment.columns)

        # Compute time until death or date date_end_study since first diagnostic code
        idx = dataset_first.death
        dataset_first.loc[idx, "observed_time"] = (
            dataset_first.loc[idx, death_date_col_name]
            - dataset_first.loc[idx, first_diagnostic_code_col_name]
        ).dt.days
        dataset_first.loc[~idx, "observed_time"] = (
            self.date_end_study
            - dataset_first.loc[~idx, first_diagnostic_code_col_name]
        ).dt.days

        data = dataset_first[
            [
                "death",
                "observed_time",
                "sex",
                "age_at_diagnostic",
                "person_id",
            ]
            + treatment_names
        ]

        n_lost = len(data.query("observed_time <= 0 "))
        if n_lost > 0:
            print(f"\n{n_lost} patients lost (diagnostic date after end of study)")
        data = data.query("observed_time > 0 ")
        return data

    def compute_cox_regression(self, data: pd.DataFrame):
        model = hazard_regression.PHReg(
            endog=data["observed_time"],
            exog=data[self.exogeneous_variables],
            status=data["death"],
            ties="efron",
        )
        result = model.fit()

        print(result.summary())
        log_HR = {
            key: value for key, value in zip(result.model.exog_names, result.params)
        }
        return log_HR

    def compute(self, stays: pd.DataFrame, shifted: bool = False, **kwargs) -> Any:
        dataset = self.preprocess(stays)
        if shifted:
            col_suffix = "_shifted"
        else:
            col_suffix = ""

        data_survival_analysis = self.get_data_to_fit(
            dataset,
            visit_number_col_name="visit_number" + col_suffix,
            birth_date_col_name="birth_date" + col_suffix,
            death_date_col_name="death_date" + col_suffix,
            first_diagnostic_code_col_name="first_diagnostic_code" + col_suffix,
        )

        return data_survival_analysis

    def indicators_estimator(self, dataset: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        data = self.compute(dataset)
        print("## Compute Cox Regression")
        log_HR = self.compute_cox_regression(data)

        ### Pseudonymized
        data_pseudo = self.compute(dataset, shifted=True)
        print("\n\n## Compute Cox Regression for pseudonymized data")
        log_HR_pseudo = self.compute_cox_regression(data_pseudo)

        beta_changes = []
        for covariate in self.exogeneous_variables:
            beta_change = np.abs(
                (log_HR[covariate] - log_HR_pseudo[covariate]) / log_HR[covariate]
            )
            beta_changes.append(beta_change)

        peseudo_effect = sum(beta_changes) / len(beta_changes)

        return dict(
            peseudo_effect=peseudo_effect,
        )

    def get_plot_class(self, **kwargs):
        return km_plot
