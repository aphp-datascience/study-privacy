import subprocess
from typing import Any, Optional

import numpy as np
import pandas as pd

from privacy.indicators.age import Age
from privacy.indicators.base import ReliabilityIndicator
from privacy.misc.utils import build_path
from privacy.plots.proportions import ProportionsPlot
from privacy.registry import registry

from loguru import logger


@registry.indicators("Cluster")
class Cluster(ReliabilityIndicator):
    name = "Cluster"

    def __init__(
        self,
        k: str = "3",
        overlap: str = "TRUE",
        step: str = "8",
        n_sample: int = 5000,
        seed: int = 35,
        seed_clustering: int = 35,
        r_env: str = "/export/home/acohen/.user_conda/miniconda/envs/r_env/lib/R/bin/Rscript",
        script_path: str = "/export/home/acohen/privacy/privacy/pipelines/clustering.R",
    ) -> None:
        super().__init__()
        self.n_sample = n_sample
        self.seed = seed
        self.r_env = r_env
        self.script_path = script_path
        self.k = k
        self.overlap = overlap
        self.step = step
        self.seed_clustering = seed_clustering
        self.patients_largest_cluster_non_pseudo = None

    def cast_to_days_from_unix(self, col):
        series = (col.view(int) / (10**9) / 86400).astype(int)

        return series

    def _add_begin_end(self, df, shifted=False):
        if shifted:
            suffix = "_shifted"
        else:
            suffix = ""

        df["visit_start_date_days_unix" + suffix] = self.cast_to_days_from_unix(
            df["visit_start_date" + suffix]
        )

        df["visit_end_date_days_unix" + suffix] = self.cast_to_days_from_unix(
            df["visit_end_date" + suffix]
        )

        df = df.merge(
            df.groupby("person_id", as_index=False).agg(
                day_start_first_stay_tmp=("visit_start_date_days_unix" + suffix, "min")
            ),
            on="person_id",
        )

        df["begin" + suffix] = (
            df["visit_start_date_days_unix" + suffix]
            - df["day_start_first_stay_tmp"]
            + 1
        )
        df["end" + suffix] = (
            df["visit_end_date_days_unix" + suffix] - df["day_start_first_stay_tmp"] + 1
        )
        df.drop(columns=["day_start_first_stay_tmp"], inplace=True)
        return df

    def lag_col_date(self, df, col_date="begin"):
        df.sort_values(["person_id", col_date], inplace=True)
        df["last_event_date"] = df[col_date].shift(fill_value=pd.NaT)

        different_person_id = df["person_id"].values[1:] != df["person_id"].values[:-1]
        different_person_id = np.append(True, different_person_id)

        df.loc[different_person_id, "last_event_date"] = pd.NaT
        return df

    def get_anniversary_event(self, year, dataset):
        birth = dataset.drop_duplicates("person_id").copy()
        delta = pd.to_timedelta(365 * year, "D")
        col_name = f"{str(year)}y"
        col_name_shifted = col_name + "_shifted"
        birth[col_name] = birth["birth_date"] + delta
        birth[col_name_shifted] = birth["birth_date_shifted"] + delta

        birth["visit_start_date"] = birth[col_name]
        birth["visit_end_date"] = birth[col_name]
        birth["visit_start_date_shifted"] = birth[col_name_shifted]
        birth["visit_end_date_shifted"] = birth[col_name_shifted]
        birth["visit_source_value"] = f"anniversary_{str(year)}"
        birth.drop(columns=[col_name, col_name_shifted], inplace=True)
        return birth

    def process(
        self,
        df,
        cols=[
            "person_id",
            "begin",
            "end",
            "care_site_short_name",
            "visit_source_value",
            "begin_shifted",
            "end_shifted",
            "birth_date",
            "birth_date_shifted",
            "visit_start_date",
            "visit_start_date_shifted",
            "visit_number",
            "visit_number_shifted",
        ],
        # anniversary_number=40,
    ):
        df = df.copy()
        # anniversary_40 = self.get_anniversary_event(40, df)
        # anniversary_50 = self.get_anniversary_event(50, df)
        anniversary_60 = self.get_anniversary_event(60, df)
        # anniversary_70 = self.get_anniversary_event(70, df)
        # anniversary_80 = self.get_anniversary_event(80, df)

        df = pd.concat(
            [
                df,
                # anniversary_40,
                # anniversary_50,
                anniversary_60,
                # anniversary_70,
                # anniversary_80,
            ],
            ignore_index=True,
        )
        df = self._add_begin_end(df)
        df = self._add_begin_end(df, shifted=True)
        return df[cols]

    def compute_agg_by_patient(self, df, shifted=False):
        if shifted:
            suffix = "_shifted"
        else:
            suffix = ""
        return df.groupby("person_id", as_index=False).agg(
            cluster=("cluster", "first"),  ## > change with pseudo
            # cluster_pseudo=("cluster_pseudo", "first"),
            mean_interval_between_stays=(
                "time_since_last_event",
                "mean",
            ),  ## > change with pseudo
            mean_stay_duration=("duration", "mean"),
            n_stays=("visit_source_value", "size"),
            first_contact=("begin" + suffix, "min"),  ## > change with pseudo
            last_contact=("end" + suffix, "max"),  ## > change with pseudo
            age_first_visit=("age" + suffix, "min"),  ## > change with pseudo
        )

    def compute_agg_by_cluster(self, df):
        return df.groupby("cluster").agg(
            n_patients=("person_id", "nunique"),
            median_interval_between_stays=(
                "mean_interval_between_stays",
                "median",
            ),  ## > change with pseudo
            median_stay_duration=("mean_stay_duration", "median"),
            median_n_stays=("n_stays", "median"),
            median_elapsed_time=(
                "elapsed_time",
                "median",
            ),  ## > change with pseudo
            median_age_first_visit=(
                "age_first_visit",
                "median",
            ),  ## > change with pseudo)
        )

    def compute(
        self, stays, patients_largest_cluster_non_pseudo: Optional[set] = None, **kwargs
    ):
        sample_person_id = (
            stays.drop_duplicates(subset="person_id")
            .sort_values("person_id")
            .person_id.sample(n=self.n_sample, random_state=self.seed)
        )

        sample = stays.loc[stays.person_id.isin(sample_person_id)].copy()
        sample.dropna(subset=["visit_start_date", "visit_end_date"], inplace=True)
        sample = self.process(sample)

        # Write data to be read from R
        path_write = build_path(__file__, "../../data/patient_traj.csv")
        sample.to_csv(path_write)
        print("### person sum", sample.person_id.sum())

        # Execute R script
        r_execution_status = subprocess.call(
            [
                self.r_env,
                "--vanilla",
                self.script_path,
                self.k,
                self.overlap,
                self.step,
                str(self.seed_clustering),
            ]
        )

        assert r_execution_status == 0

        # Import results
        clustering = pd.read_csv(
            build_path(__file__, "../../data/clustering_shifted.csv"),
            names=["person_id", "cluster"],
            header=0,
        )

        # Delete anniversary events
        sample = sample.loc[~sample.visit_source_value.str.contains("anniversary_")]

        # Add cluster to the sample
        sample = sample.merge(
            clustering, on="person_id", how="left", validate="many_to_one"
        )

        # Compute duration of stays
        sample["duration"] = sample["end_shifted"] - sample["begin_shifted"]

        # Lag column & compute time interval since last stay event
        sample = self.lag_col_date(sample, col_date="end")
        sample["time_since_last_event"] = (
            sample["begin_shifted"] - sample["last_event_date"]
        )

        # Compute age
        sample["age_shifted"] = Age().compute(
            sample, shifted=True
        )

        # Compute aggregations by patient
        patient_aggregations = self.compute_agg_by_patient(sample, shifted=True)

        # Compute elapsed time between first and last contact of the patient
        patient_aggregations["elapsed_time"] = (
            patient_aggregations["last_contact"] - patient_aggregations["first_contact"]
        )

        # Compute descriptors of each cluster
        cluster_description = self.compute_agg_by_cluster(patient_aggregations)

        # Retrieve patients of the largest cluster
        idx = cluster_description.n_patients.argmax()
        largest_cluster = cluster_description.iloc[idx].name
        sample["largest_cluster"] = sample.cluster == largest_cluster
        patients_largest_cluster = set(
            sample.loc[sample.largest_cluster].person_id.drop_duplicates()
        )

        if patients_largest_cluster_non_pseudo:
            indicator = 1 - len(
                patients_largest_cluster_non_pseudo.intersection(
                    patients_largest_cluster
                )
            ) / len(patients_largest_cluster_non_pseudo)

        else:
            indicator = 0

        return (
            cluster_description.sort_values("n_patients", ascending=False),
            sample,
            patients_largest_cluster,
            indicator,
        )

    def indicators_estimator(
        self,
        stays: pd.DataFrame,
        pseudonymization_algorithm: str = "NoPseudonymizer",
        return_cluster_description: bool = False,
        **kwargs,
    ):
        """

        Parameters
        ----------
        stays : pd.DataFrame
            Pseudonymized stays
        """
        if pseudonymization_algorithm == "NoPseudonymizer":
            (
                cluster_description,
                _,
                patients_largest_cluster_non_pseudo,
                indicator,
            ) = self.compute(stays)

            self.patients_largest_cluster_non_pseudo = (
                patients_largest_cluster_non_pseudo
            )
        else:
            if self.patients_largest_cluster_non_pseudo is None:
                logger.warning("patients_largest_cluster_non_pseudo is None")
            (
                cluster_description,
                _,
                _,
                indicator,
            ) = self.compute(
                stays,
                patients_largest_cluster_non_pseudo=self.patients_largest_cluster_non_pseudo,
            )

        results_indicator = {"cluster_indicator": indicator}
        if return_cluster_description:
            results_indicator["cluster_description"] = cluster_description

        return results_indicator

    def get_plot_class(self, **kwargs):
        pc = ProportionsPlot(
            ylabel="Proportion of individuals who switch clusters after pseudonymization",
            **kwargs,
        )
        return pc
