from typing import Iterable, List

import numpy as np
import pandas as pd
from confit import Config
from loguru import logger

from privacy import registry
from privacy.indicators.age import Age
from privacy.misc.utils import DataModule
from privacy.pipelines.uniqueness import pipe_uniqueness


class SuppMaterialTablesGenerator:
    def __init__(
        self,
        bins_stay_length: List[int] = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            20,
            40,
            80,
            150,
        ],
        n_test_uniqueness: int = 50,
        seed: int = 55,
        tables: Iterable[str] = [
            "frequency_stays",
            "uniqueness_hospital",
            "uniqueness_stay_length",
            "uniqueness_age_gender",
        ],
        number_of_stays_max: int = 10,
        batch_size: int = 50,
    ) -> None:
        self.bins_stay_length = bins_stay_length
        self.n_test_uniqueness = n_test_uniqueness
        self.seed = seed
        self.tables = tables
        self.number_of_stays_max = number_of_stays_max
        self.batch_size = batch_size

    def compute_bins(
        self, df, column, nan_label="90+", interval=10, max_bin_value=100, bins=None
    ):
        df = df.query(f"{column} >=0 & {column}.notna() ").copy()
        if bins is None:
            bins = np.arange(0, max_bin_value, interval)
        labels = [
            f"[{left}-{int(right) - 1}]" if (left != (right - 1)) else f"{left}"
            for left, right in zip(bins[:-1], bins[1:])
        ]
        df[f"{column}_bins"] = pd.cut(
            df[column],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False,
        )

        df[f"{column}_bins"] = df[f"{column}_bins"].cat.add_categories(nan_label)
        df[f"{column}_bins"] = df[f"{column}_bins"].fillna(nan_label)
        return df

    def pipe_uniqueness_by_bin(
        self,
        df: pd.DataFrame,
        gb_cols: List[str],
        overall_cohort: pd.DataFrame,
        overall_stays: pd.DataFrame,
        n: int = 50,
        seed: int = 55,
    ):
        count_by_bin = df.groupby(gb_cols, as_index=False).size()
        count_by_bin = count_by_bin.query(f"size>={n}").drop(columns=["size"])

        df = df.merge(count_by_bin, on=gb_cols, how="inner", validate="many_to_one")

        test_patients = (
            df[
                [
                    "person_id",
                ]
                + gb_cols
            ]
            .groupby(gb_cols)
            .sample(n, random_state=seed)
        )

        results = pipe_uniqueness(
            overall_cohort=overall_cohort,
            overall_stays=overall_stays,
            patients_to_check=test_patients.person_id.to_list(),
            pseudonymization_algorithm="NoPseudonymizer",
            attack_knowledge=[
                "visit_start_date",
                "visit_end_date",
                "birth_date",
                "death_date",
                "gender",
                "hospital",
            ],
            return_tables=True,
            batch_size=self.batch_size,
            **dict(
                low_general=0,
                high_general=0,
                low_birth_date=0,
                high_birth_date=0,
            ),
        )

        uniqueness = results["result"].to_pandas()
        test_patients_w_info = test_patients.merge(uniqueness, on="person_id")

        uniqueness_gp = test_patients_w_info.groupby(gb_cols).unique.mean()
        uniqueness_gp.name = "uniqueness"

        uniqueness_gp = pd.DataFrame(uniqueness_gp)

        counts_bucket = df.groupby(gb_cols)["person_id"].count()
        counts_bucket.name = "count"
        counts_bucket = pd.DataFrame(counts_bucket)

        result = counts_bucket.merge(uniqueness_gp, left_index=True, right_index=True)

        return result

    def __call__(self, data):
        results = {}
        logger.info(f"Computing tables for {self.tables}")

        # Get Stays and cohort
        stays = data.stays("all_population")
        cohort = data.cohort("all_population")

        # first stays
        first_stays = stays.query("visit_number == 1")
        first_stays = first_stays.merge(cohort, on="person_id", validate="many_to_one")

        if "uniqueness_stay_length" in self.tables:
            logger.info("--- ### COMPUTING uniqueness_stay_length ### ---")
            # Stay length
            first_stays["duration"] = (
                first_stays["visit_end_date"] - first_stays["visit_start_date"]
            )
            first_stays["duration"] = first_stays.duration.dt.days

            # Bin stay length
            first_stays_duration = self.compute_bins(
                first_stays,
                column="duration",
                bins=self.bins_stay_length,
                nan_label="+" + str(self.bins_stay_length[-1]) + "D",
            )

            # Compute uniqueness by stay length (duration) bin
            uniqueness_stay_length = self.pipe_uniqueness_by_bin(
                first_stays_duration,
                [
                    "duration_bins",
                ],
                overall_cohort=cohort,
                overall_stays=stays,
                n=self.n_test_uniqueness,
                seed=self.seed,
            )
            results["uniqueness_stay_length"] = uniqueness_stay_length

        if "uniqueness_age_gender" in self.tables:
            logger.info("--- ### COMPUTING uniqueness_age_gender ### ---")
            # Compute age
            first_stays["age"] = Age._compute_age(
                first_stays,
            )
            # Bin age
            first_stays_age = self.compute_bins(
                first_stays,
                column="age",
            )

            # Keep only patients w gender
            first_stays_age = first_stays_age.loc[
                first_stays_age["gender_source_value"].isin(["m", "f"])
            ]

            # Compute uniqueness for each age & gender bin
            uniqueness_age_gender = self.pipe_uniqueness_by_bin(
                first_stays_age,
                ["age_bins", "gender_source_value"],
                overall_cohort=cohort,
                overall_stays=stays,
                n=self.n_test_uniqueness,
                seed=self.seed,
            )
            results["uniqueness_age_gender"] = uniqueness_age_gender

        if "uniqueness_hospital" in self.tables:
            logger.info("--- ### COMPUTING uniqueness_hospital ### ---")
            # Compute uniqueness by hospital
            uniqueness_hospital = self.pipe_uniqueness_by_bin(
                first_stays,
                [
                    "care_site_short_name",
                ],
                overall_cohort=cohort,
                overall_stays=stays,
                n=self.n_test_uniqueness,
                seed=self.seed,
            )

            # Sort values
            uniqueness_hospital.sort_values("count", inplace=True, ascending=False)

            results["uniqueness_hospital"] = uniqueness_hospital

        if "frequency_stays" in self.tables:
            logger.info("--- ### COMPUTING frequency_stays ### ---")
            # Count number of stays by patient
            count_by_patient = stays.groupby("person_id", as_index=False).size()
            count_by_patient.rename(columns={"size": "number_of_stays"}, inplace=True)
            count_by_patient = count_by_patient.loc[
                count_by_patient.number_of_stays <= self.number_of_stays_max
            ]
            stays_w_count = stays.merge(
                count_by_patient, on="person_id", validate="many_to_one", how="inner"
            )

            # Compute uniqueness by number of stays
            uniqueness_n_stays = self.pipe_uniqueness_by_bin(
                stays_w_count,
                [
                    "number_of_stays",
                ],
                overall_cohort=cohort,
                overall_stays=stays,
                n=self.n_test_uniqueness,
                seed=self.seed,
            )
            uniqueness_n_stays.drop(
                columns=[
                    "count",
                ],
                inplace=True,
            )

            frequency_stays = count_by_patient.groupby(
                "number_of_stays", as_index=True
            ).size()
            frequency_stays.name = "count"
            frequency_stays = pd.DataFrame(frequency_stays)

            result_freq_stays = frequency_stays.merge(
                uniqueness_n_stays,
                left_index=True,
                right_index=True,
                validate="one_to_one",
            )

            result_freq_stays.sort_index(inplace=True)
            results["frequency_stays"] = result_freq_stays

        return results
