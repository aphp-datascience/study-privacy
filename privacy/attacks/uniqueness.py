import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import polars as pl
from loguru import logger

from privacy.misc.assignement import Assignement
from privacy.misc.data_wrangling import pandas_to_polars_timedelta


class Uniqueness:
    def __init__(
        self,
        pseudonymization_process: Dict[str, Any],
        attack_knowledge: List[str] = [
            "visit_start_date",
            "visit_end_date",
            "birth_date",
            "death_date",
            "gender",
            "hospital",
        ],
        **kwargs,
    ) -> None:
        """Class to evaluate the uniqueness of a patient given a cohort, the attacker knowledge and a pseudonymization process

        Parameters
        ----------
        pseudonymization_process : Dict[str, Any]
            Process used to pseudonimize records. The algorithm name and the shifts are expected.
        attack_knowledge : List[str], optional
            Knowledge of the attackers, by default [ "visit_start_date", "visit_end_date", "birth_date", "death_date", "gender","hospital" ]
        """
        self.pseudonymization_process = pseudonymization_process
        self.attack_knowledge = set(attack_knowledge)
        pass

    @staticmethod
    def hash_patients(
        dataset: pl.DataFrame,
        attack_knowledge: List[str],
        pseudonymization_algorithm: str,
    ) -> pl.DataFrame:
        """Make a hash to find candidates for each patient.

        if:
            `pseudonymization_algorithm` in {"BasePseudonymizer","BirthPseudonymizer"}
            and the attacker knows `visit_start_date` or  `visit_end_date`. So the
            hash = `gender + hospital sequence + vital status`.

        elif:
            `pseudonymization_algorithm` in {"StayPseudonymizer",}
            and the attacker knows `visit_start_date` or  `visit_end_date`. So the
            hash = `gender + alphabetical sorted hospital sequence + vital status`.

        else:
            hash = `gender + vital status`.

        Parameters
        ----------
        dataset : pl.DataFrame
        attack_knowledge : List[str]
        pseudonymization_algorithm : str
            One of {"BasePseudonymizer","BirthPseudonymizer","StayPseudonymizer"}

        Returns
        -------
        pl.DataFrame
            df with `person_id` & `hash`
        """

        assert "gender_source_value" in dataset.columns

        cols_visit_date = set(
            [
                "visit_start_date",
                "visit_end_date",
            ]
        )
        hospital_sequence_knowledge = "hospital" in attack_knowledge

        assert set(["gender", "death_date"]).issubset(attack_knowledge)

        dataset = dataset.sort(
            cols_visit_date,
        )
        dataset = dataset.with_columns(
            [
                (pl.col("death_date").is_not_null()).alias("death"),
            ]
        )

        sequences = dataset.groupby("person_id").agg(
            [
                pl.col("care_site_short_name").alias("sequence"),
                pl.col("death").first().alias("death"),
                pl.col("gender_source_value").first().alias("gender"),
            ]
        )

        sequences = sequences.with_columns(
            [
                (pl.col("sequence").arr.sort()).alias("sequence_sorted"),
                pl.col("sequence").arr.lengths().alias("sequence_length"),
            ]
        )

        if hospital_sequence_knowledge and (
            pseudonymization_algorithm
            in {
                "BasePseudonymizer",
                "BirthPseudonymizer",
            }
        ):
            sequences = sequences.with_columns(
                pl.col("sequence").arr.join("_").alias("sequence_cat"),
            )

            sequences = sequences.with_columns(
                (
                    pl.col("gender")
                    + "_"
                    + pl.col("sequence_cat")
                    + "_"
                    + pl.col("death")
                ).alias("hash")
            )
        elif hospital_sequence_knowledge and (
            pseudonymization_algorithm == "StayPseudonymizer"
        ):
            sequences = sequences.with_columns(
                pl.col("sequence_sorted").arr.join("_").alias("sequence_cat_sorted"),
            )
            sequences = sequences.with_columns(
                (
                    pl.col("gender")
                    + "_"
                    + pl.col("sequence_cat_sorted")
                    + "_"
                    + pl.col("death")
                ).alias("hash"),
            )
        else:
            sequences = sequences.with_columns(
                (pl.col("gender") + "_" + pl.col("death")).alias("hash"),
            )

        # drop useless columns
        sequences = sequences.drop(
            list(
                set(sequences.columns).intersection(
                    set(
                        [
                            "sequence",
                            "death",
                            "gender",
                            "sequence_cat",
                            "sequence_cat_sorted",
                            "sequence_sorted",
                        ]
                    )
                )
            )
        )

        # Replace Hash by an int to reduce memory usage
        hash_correspondance = sequences.select("hash").unique()
        hash_correspondance = hash_correspondance.with_row_count(name="hash_int")
        sequences = sequences.join(hash_correspondance, on="hash", how="inner")
        sequences = sequences.drop(columns=["hash"])
        sequences = sequences.rename({"hash_int": "hash"})
        return sequences

    @staticmethod
    def compute_difference_between_a_b(
        df: pl.DataFrame,
        col_name_result: str,
        a: str,
        b: str,
    ):
        """Compute the difference between column `a` and `b`.
            col_name_result =  a - b

        Parameters
        ----------
        df : pd.DataFrame
        col_name_result : str
            column name to save the result
        a : str
            column name
        b : str
            column name
        """
        return df.with_columns((pl.col(a) - pl.col(b)).alias(col_name_result))

    @staticmethod
    def compute_shift_between_candidate_patient(dataset: pl.DataFrame) -> pl.DataFrame:
        """Compute time differnece between patient and candidate for
        the columns {`birth_date`,`death_date`, `visit_start_date`}

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pl.DataFrame
        """
        dataset = Uniqueness.compute_difference_between_a_b(
            dataset,
            "shift_candidate_patient_birth_date",
            "birth_date_shifted_candidate",
            "birth_date",
        )
        dataset = Uniqueness.compute_difference_between_a_b(
            dataset,
            "shift_candidate_patient_death_date",
            "death_date_shifted_candidate",
            "death_date",
        )

        dataset = Uniqueness.compute_difference_between_a_b(
            dataset,
            "shift_candidate_patient_visit_start_date",
            "visit_start_date_shifted_candidate",
            "visit_start_date",
        )
        return dataset

    def merge_with_candidates(
        self, stays_patients_to_check: pl.DataFrame, stays: pl.DataFrame
    ) -> pl.DataFrame:
        """Merge stays of patients to check with stays of all possible candidates.

        Parameters
        ----------
        stays_patients_to_check : pd.DataFrame
            df with the stays of patients to verify the uniqueness
        stays : pd.DataFrame
            df with the stays of all patients of the cohort

        Returns
        -------
        pd.DataFrame
            columns coming from candidate df have the suffix `_candidate`. Ex: {column_name}_candidate
        """

        hospital_sequence_knowledge = "hospital" in self.attack_knowledge

        # if != StayPseudo et pas connaisance de visit_date
        # >> on peut pas utiliser visit_number ? (logique)
        # >> on peut garder qu'une visite par patient (memoire)
        # Si on connait pas les infos des dates, on connait que les infos niveau patient !
        if (self.pseudonymization_process["algorithm"] != "StayPseudonymizer") and (
            hospital_sequence_knowledge
        ):
            logger.debug("Merge by hash + visit_number")
            left_on = ["hash", "visit_number"]
            right_on = ["hash", "visit_number_shifted_candidate"]

        elif not hospital_sequence_knowledge:
            logger.debug("Merge only by hash and drop rows (keep one stay per patient)")
            left_on = [
                "hash",
            ]
            right_on = [
                "hash",
            ]

            stays = stays.unique(subset=["person_id"])
            stays_patients_to_check = stays_patients_to_check.unique(
                subset=["person_id"]
            )

        else:
            logger.debug("Merge only by hash")
            left_on = [
                "hash",
            ]
            right_on = [
                "hash",
            ]

        stays = stays.select(
            [
                "person_id",
                "hash",
                "visit_start_date_shifted",
                "visit_end_date_shifted",
                "birth_date_shifted",
                "death_date_shifted",
                "duration",
                "visit_number_shifted",
                "care_site_id",
            ]
        )

        stays = stays.rename(
            {
                "visit_start_date_shifted": "visit_start_date_shifted_candidate",
                "visit_end_date_shifted": "visit_end_date_shifted_candidate",
                "birth_date_shifted": "birth_date_shifted_candidate",
                "death_date_shifted": "death_date_shifted_candidate",
                "visit_number_shifted": "visit_number_shifted_candidate",
            }
        )

        stays_patient_candidates = stays_patients_to_check.join(
            stays,
            left_on=left_on,
            right_on=right_on,
            how="left",
            suffix="_candidate",
        )

        return stays_patient_candidates

    @staticmethod
    def distinct_by_date(
        df: pd.DataFrame,
        col_name: str,
        low: pd.Timedelta,
        high: pd.Timedelta,
        alias: str,
    ) -> pl.DataFrame:
        """Returns `True` if stays are not compatible by date intervals.
        `{col_name}_shifted_candidate` should be between `{col_name}` +- dt

        Parameters
        ----------
        df : pd.DataFrame
        col_name : str
            column name to check
        low : pd.Timedelta
            lower bound of interval
        high : pd.Timedelta
            upper bound of interval
        alias : str
            name output col

        Returns
        -------
        pl.DataFrame
            same df with an extra boolean column named `alias`
        """
        low = pandas_to_polars_timedelta(low)
        high = pandas_to_polars_timedelta(high)
        col_name_candidate = f"{col_name}_shifted_candidate"
        df = df.with_columns(
            (
                (  # Date_candidate between interval and both not null
                    pl.col(col_name_candidate).is_between(
                        (pl.col(col_name) + low), (pl.col(col_name) + high)
                    )
                )
                .and_(
                    pl.col(col_name).is_not_null(),
                    pl.col(col_name_candidate).is_not_null(),
                )
                .or_(  # Both null
                    (pl.col(col_name).is_null()).and_(
                        pl.col(col_name_candidate).is_null()
                    )
                )
            )
            .is_not(
                # Negation of everything
            )
            .alias(alias),
        )

        return df

    @staticmethod
    def distinct_by_duration(
        df: pl.DataFrame, alias="CONDITION_DURATION"
    ) -> pl.DataFrame:
        """Returns `True` if stay duration of patient and stay duration of candidate
        are not equal. Else returns `False`

        Parameters
        ----------
        df : pl.DataFrame
            a df with columns `duration` and `duration_candidate`

        Returns
        -------
        pl.DataFrame
            same df with an extra boolean column named `alias`
        """
        df = df.with_columns(
            pl.col("duration").ne(pl.col("duration_candidate")).alias(alias),
        )

        return df

    @staticmethod
    def distinct_by_hospital(
        df: pl.DataFrame, alias: str = "CONDITION_HOSPITAL"
    ) -> pl.DataFrame:
        """Returns `True` if hospital of patient stay is different from candidate stay.
        Else returns `False`

        Parameters
        ----------
        df : pl.DataFrame
            a df with columns `care_site_id` and `care_site_id_candidate`
        alias : str
            column name for the output

        Returns
        -------
        pl.DataFrame
            same df with an extra boolean column named `alias`
        """

        df = df.with_columns(
            pl.col("care_site_id").ne(pl.col("care_site_id_candidate")).alias(alias),
        )
        return df

    @staticmethod
    def check_uniqueness_stay_level(
        df: pl.DataFrame,
        pseudonymization_process: Dict[str, Any],
        attack_knowledge: List[str] = [
            "visit_start_date",
            "visit_end_date",
            "birth_date",
            "death_date",
            "gender",
            "hospital",
        ],
    ) -> pl.DataFrame:
        """Check compatibility for each stay of possible pairs.

        It adds the column `distinct`. It takes value `True`
        if given the pseudonymization process and the previous attacker
        knwoledge we can differentiate between the two stays.

        Parameters
        ----------
        df : pl.DataFrame
        pseudonymization_process : Dict[str, Any]
            dictionary with shifts information (general and birth)
        attack_knowledge : List[str], optional
            list of previous knowledge of the attacker,
            by default [ "visit_start_date", "visit_end_date", "birth_date", "death_date", "gender","hospital" ]

        Returns
        -------
        pl.DataFrame
        with the extra column `distinct`
        """
        # assert df.index.is_unique

        df = df.with_columns(pl.lit(True).alias("distinct"))
        attack_knowledge = set(attack_knowledge)
        cols_visit_date = set(
            [
                "visit_start_date",
                "visit_end_date",
            ]
        )

        assert set(
            [
                "gender",
                "birth_date",
                "death_date",
            ]
        ).issubset(attack_knowledge)
        CONDITIONS = []

        # Duration
        if cols_visit_date.issubset(attack_knowledge):
            df = Uniqueness.distinct_by_duration(df, alias="CONDITION_DURATION")
            CONDITIONS.append("CONDITION_DURATION")

        # Check birth date
        if "birth_date" in attack_knowledge:
            df = Uniqueness.distinct_by_date(
                df,
                "birth_date",
                low=pseudonymization_process["shifts"]["birth"]["low"],
                high=pseudonymization_process["shifts"]["birth"]["high"],
                alias="CONDITION_BIRTH_DATE",
            )
            CONDITIONS.append("CONDITION_BIRTH_DATE")

        if "death_date" in attack_knowledge:
            df = Uniqueness.distinct_by_date(
                df,
                "death_date",
                low=pseudonymization_process["shifts"]["general"]["low"],
                high=pseudonymization_process["shifts"]["general"]["high"],
                alias="CONDITION_DEATH_DATE",
            )
            CONDITIONS.append("CONDITION_DEATH_DATE")

        if "visit_start_date" in attack_knowledge:
            df = Uniqueness.distinct_by_date(
                df,
                "visit_start_date",
                low=pseudonymization_process["shifts"]["general"]["low"],
                high=pseudonymization_process["shifts"]["general"]["high"],
                alias="CONDITION_START_DATE",
            )
            CONDITIONS.append("CONDITION_START_DATE")

        if "visit_end_date" in attack_knowledge:
            df = Uniqueness.distinct_by_date(
                df,
                "visit_end_date",
                low=pseudonymization_process["shifts"]["general"]["low"],
                high=pseudonymization_process["shifts"]["general"]["high"],
                alias="CONDITION_END_DATE",
            )
            CONDITIONS.append("CONDITION_END_DATE")

        # Hospital
        # It could know the hospital only without knowing stay dates
        if "hospital" in attack_knowledge:
            df = Uniqueness.distinct_by_hospital(df, alias="CONDITION_HOSPITAL")
            CONDITIONS.append("CONDITION_HOSPITAL")

        # Reduce conditions
        df = df.with_columns(
            pl.when(pl.any(CONDITIONS))
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("distinct")
        )

        return df

    def drop_cols(
        self,
        df,
        cols_keep=[
            "person_id",
            "person_id_candidate",
            "distinct",
            "hash",
            "sequence_length",
            "shift_candidate_patient_birth_date",
            "shift_candidate_patient_visit_start_date",
            "shift_candidate_patient_death_date",
            "visit_number",
            "visit_number_shifted_candidate",
        ],
    ):
        cols_to_drop = list(set(df.columns).difference(set(cols_keep)))
        df = df.drop(columns=cols_to_drop)
        return df

    def aggregate_couple_information(
        self,
        stays_patient_candidates: pl.DataFrame,
    ) -> pl.DataFrame:
        """Aggregate information for each couple (patient, candidate)

        Parameters
        ----------
        stays_patient_candidates : pl.DataFrame
            Aligned stays of patient & candidate

        Returns
        -------
        pl.DataFrame
        """
        if self.pseudonymization_process["algorithm"] != "StayPseudonymizer":
            patient_candidates = stays_patient_candidates.groupby(
                ["person_id", "person_id_candidate"]
            ).agg(
                distinct=pl.any("distinct"),
                n_stays_common=pl.count(),
                n_stays_patient=pl.first("sequence_length"),
                n_birth_date_shifts=pl.n_unique("shift_candidate_patient_birth_date"),
                n_visit_start_date_shifts=pl.n_unique(
                    "shift_candidate_patient_visit_start_date"
                ),
                birth_date_shift=pl.first("shift_candidate_patient_birth_date"),
                death_date_shift=pl.first("shift_candidate_patient_death_date"),
                first_shift_candidate_patient_visit_start_date=pl.first(
                    "shift_candidate_patient_visit_start_date"
                ),
            )
        else:
            patient_candidates = (
                stays_patient_candidates.filter(pl.col("distinct").is_not())
                .groupby(
                    ["person_id", "person_id_candidate"], maintain_order=True
                )  # should I set to True (slower) or can i leave False (default) ? TODO
                .agg(
                    distinct=pl.any(
                        "distinct"
                    ),  # For compatibility but always False because of the filter.
                    patient_number_of_precompatible_stays=pl.n_unique("visit_number"),
                    candidate_number_of_precompatible_stays=pl.n_unique(
                        "visit_number_shifted_candidate"
                    ),
                    n_stays_patient=pl.first("sequence_length"),
                    visit_number=pl.col("visit_number"),
                    visit_number_shifted_candidate=pl.col(
                        "visit_number_shifted_candidate"
                    ),
                )
            )

        return patient_candidates

    def check_uniqueness_patient_level(
        self,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Check temporal consistency between all stays of candidate and patient.
        It adds the column `temporal_inconsistency`. It takes value `True` if
        there is not temporal coherence. All stays should be compatible.

        Parameters
        ----------
        df : pl.DataFrame

        Returns
        -------
        pl.DataFrame
        """
        algorithm = self.pseudonymization_process["algorithm"]

        visit_date_cols_knowledge = not set(
            [
                "visit_start_date",
                "visit_end_date",
            ]
        ).isdisjoint(self.attack_knowledge)

        vital_status_cols_knowledge = set(
            [
                "birth_date",
                "death_date",
            ]
        ).issubset(self.attack_knowledge)

        low_birth_parameter = (
            self.pseudonymization_process["shifts"]["birth"]["low"]
            - self.pseudonymization_process["shifts"]["general"]["low"]
        )
        high_birth_parameter = (
            self.pseudonymization_process["shifts"]["birth"]["high"]
            - self.pseudonymization_process["shifts"]["general"]["high"]
        )

        if visit_date_cols_knowledge:
            if algorithm == "BasePseudonymizer":
                # Algo 1
                condition_algo_base = (pl.col("distinct").is_not()).and_(
                    pl.col("n_visit_start_date_shifts") == 1,
                    pl.col("birth_date_shift")
                    == pl.col("first_shift_candidate_patient_visit_start_date"),
                    (
                        (
                            pl.col("death_date_shift")
                            == pl.col("first_shift_candidate_patient_visit_start_date")
                        ).or_(pl.col("death_date_shift").is_null())
                    ),
                    pl.col("n_stays_patient") == pl.col("n_stays_common"),
                )

                df = df.with_columns(
                    pl.when(condition_algo_base)
                    .then(pl.col("distinct"))
                    .otherwise(pl.lit(True))
                    .alias("temporal_inconsistency")
                )

            if algorithm == "BirthPseudonymizer":
                condition_algo_birth = (pl.col("distinct").is_not()).and_(
                    # Only one stay shift
                    pl.col("n_visit_start_date_shifts") == 1,
                    # Death shift = stay shift
                    (
                        (
                            pl.col("death_date_shift")
                            == pl.col("first_shift_candidate_patient_visit_start_date")
                        ).or_(pl.col("death_date_shift").is_null())
                    ),
                    # Patient and candidate
                    pl.col("n_stays_patient") == pl.col("n_stays_common"),
                    # Birth date shift is between low and high once stay's shift is kwnown
                    pl.col("birth_date_shift").is_between(
                        (
                            pl.col("first_shift_candidate_patient_visit_start_date")
                            + low_birth_parameter
                        ),
                        (
                            pl.col("first_shift_candidate_patient_visit_start_date")
                            + high_birth_parameter
                        ),
                    ),
                )

                df = df.with_columns(
                    pl.when(condition_algo_birth)
                    .then(pl.col("distinct"))
                    .otherwise(pl.lit(True))
                    .alias("temporal_inconsistency")
                )

            if algorithm == "StayPseudonymizer":
                # Algo 3
                condition_algo_stay = (pl.col("distinct").is_not()).and_(
                    pl.col("n_stays_patient")
                    == pl.col("candidate_number_of_precompatible_stays"),
                    pl.col("n_stays_patient")
                    == pl.col("patient_number_of_precompatible_stays"),
                )

                df = df.with_columns(
                    pl.when(condition_algo_stay)
                    .then(pl.col("distinct"))
                    .otherwise(pl.lit(True))
                    .alias("temporal_inconsistency")
                )

                # Assignement check
                # For n_stays_patient <=2, previous verification is sufficient.
                condition_assignement_check = (
                    pl.col("temporal_inconsistency").is_not()
                ).and_(pl.col("n_stays_patient") > 2)

                n = df.filter(condition_assignement_check).shape[0]
                logger.debug(f"Couples to check Combinatorial Assignement {n}")

                df = df.with_columns(
                    pl.when(condition_assignement_check)
                    .then(
                        pl.struct(
                            [
                                "visit_number",
                                "visit_number_shifted_candidate",
                                "n_stays_patient",
                            ]
                        ).apply(
                            lambda x: not Assignement(
                                patient_stay_number=x["visit_number"],
                                candidate_stay_number=x[
                                    "visit_number_shifted_candidate"
                                ],
                                n_stays_patient=x["n_stays_patient"],
                            ).is_compatible()
                        )
                    )
                    .otherwise(pl.col("temporal_inconsistency"))
                    .alias("temporal_inconsistency")
                )

        elif (vital_status_cols_knowledge) and (
            algorithm == "BasePseudonymizer"
        ):  # (visit_date_cols_knowledge is False)
            condition_algo_base = (pl.col("distinct").is_not()).and_(
                (pl.col("death_date_shift") == pl.col("birth_date_shift")).or_(
                    pl.col("death_date_shift").is_null()
                )
            )

            df = df.with_columns(
                pl.when(condition_algo_base)
                .then(pl.col("distinct"))
                .otherwise(pl.lit(True))
                .alias("temporal_inconsistency")
            )
        elif (vital_status_cols_knowledge) and (algorithm == "BirthPseudonymizer"):
            condition_algo_birth = (pl.col("distinct").is_not()).and_(
                # Birth date shift is between low and high once death's shift is kwnown
                pl.col("birth_date_shift")
                .is_between(
                    (pl.col("death_date_shift") + low_birth_parameter),
                    (pl.col("death_date_shift") + high_birth_parameter),
                )
                .or_(pl.col("death_date_shift").is_null())
            )

            df = df.with_columns(
                pl.when(condition_algo_birth)
                .then(pl.col("distinct"))
                .otherwise(pl.lit(True))
                .alias("temporal_inconsistency")
            )
        else:  # (visit_date_cols_knowledge and  vital_status_cols_knowledge are False)
            df = df.with_columns(pl.col("distinct").alias("temporal_inconsistency"))

        return df

    @staticmethod
    def aggregate_patient_information(patient_candidates: pl.DataFrame) -> pl.DataFrame:
        """Aggregate results for each patient.
        It returns a df with the column `unique` if there is only one candidate that matches.

        Parameters
        ----------
        patient_candidates : pd.DataFrame
            aggregated df of patient & candidate

        Returns
        -------
        pl.DataFrame
        """
        result = patient_candidates.groupby("person_id").agg(
            candidates_w_temporal_inconsistency=pl.sum("temporal_inconsistency"),
            total_candidates=pl.count("temporal_inconsistency"),
        )

        result = result.with_columns(
            (
                pl.col("total_candidates")
                - pl.col("candidates_w_temporal_inconsistency")
            ).alias("remainder")
        )

        result = result.with_columns((pl.col("remainder") == 1).alias("unique"))

        result = result.drop(
            [
                "remainder",
                "total_candidates",
                "candidates_w_temporal_inconsistency",
            ]
        )
        return result

    def _read_data(
        self, __obj: Union[pd.DataFrame, str, Path]
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        if isinstance(__obj, pd.DataFrame):
            return pl.from_pandas(__obj)

        if isinstance(__obj, pl.DataFrame):
            return __obj

        elif isinstance(__obj, (str, Path)):
            return pl.read_parquet(__obj, low_memory=True)
        else:
            raise NotImplementedError

    def process_chunk(self, stays_pl, patients_chunk, return_stays):
        # Select columns to minimize memory usage
        stays_patients_to_check = stays_pl.filter(
            pl.col("person_id").is_in(patients_chunk)
        ).select(
            [
                "care_site_id",
                "person_id",
                "visit_start_date",
                "visit_end_date",
                "visit_number",
                "birth_date",
                "death_date",
                "duration",
                "hash",
                "sequence_length",
            ]
        )

        if not isinstance(stays_pl, pl.LazyFrame):
            logger.debug(
                f"Starting merge with candidates. {len(stays_patients_to_check)} patient stays with {len(stays_pl)} stays"
            )
        # Merge with candidates
        stays_patient_candidates = self.merge_with_candidates(
            stays_patients_to_check=stays_patients_to_check, stays=stays_pl
        )

        if not isinstance(stays_patient_candidates, pl.LazyFrame):
            logger.debug(f"stays_patient_candidates : {len(stays_patient_candidates)}")

        # Check unicity of stays
        stays_patient_candidates = self.check_uniqueness_stay_level(
            stays_patient_candidates,
            pseudonymization_process=self.pseudonymization_process,
            attack_knowledge=self.attack_knowledge,
        )

        logger.debug(f"check_uniqueness_stay_level : ok")

        # compute shifts for each stay individually
        stays_patient_candidates = self.compute_shift_between_candidate_patient(
            stays_patient_candidates
        )

        logger.debug(f"compute_shift_between_candidate_patient : ok ")

        # Drop cols
        stays_patient_candidates = self.drop_cols(stays_patient_candidates)

        # Aggregate information for each couple
        patient_candidates = self.aggregate_couple_information(
            stays_patient_candidates=stays_patient_candidates,
        )

        logger.debug(f"patient_candidates : ok")

        # Check temporal & sequence consistency (in function of pseudonymization algorithm)
        patient_candidates = self.check_uniqueness_patient_level(patient_candidates)

        logger.debug(f"temporal_consistency : ok")

        # Aggregate information for each patient
        result = self.aggregate_patient_information(
            patient_candidates=patient_candidates
        )
        if return_stays:
            return result, patient_candidates, stays_patient_candidates
        else:
            return result, None, None

    def __call__(
        self,
        dataset: Union[pd.DataFrame, str, Path],
        patients_to_check: Optional[pd.Series] = None,
        cohort: Optional[Union[pd.DataFrame, str, Path]] = None,
        sample_size: Optional[int] = 500,
        seed: int = 55,
        pre_hashed: bool = False,
        batch_size: Optional[int] = None,
        return_stays: bool = False,
    ) -> Any:
        """

        Parameters
        ----------

        dataset : Union[pd.DataFrame, str, Path]
            Pseudonymized dataset (stays granularity) with patient information (birth_date, death_date, gender)
        patients_to_check : Optional[pd.Series], optional
            pd.Series of person_id to check uniqueness, by default None
        cohort : Optional[Union[pd.DataFrame, str, Path]]
                cohort to sample patients if patients_to_check is None
                by default None
        sample_size : Optional[int], optional
            sample size of cohort to verify if patients_to_check is None, by default 500
        seed : int, optional
            seed for sample, by default 55
        pre_hashed : bool, optional
            whether to avoid hash computing, by default False

        Returns
        -------
        (pd.DataFrame,pd.DataFrame,pd.DataFrame)
            result, patient_candidates, stays_patient_candidates
        """
        logger.info(f"--- ### START UNIQUENESS COMPUTING ### ---")
        logger.debug(f"Attack knowledge: {self.attack_knowledge}")
        logger.debug(
            f"Pseudonimization algorithm: {self.pseudonymization_process['algorithm']}"
        )

        if cohort is not None:
            cohort_pl = self._read_data(cohort)
            if not isinstance(cohort_pl, pl.LazyFrame):
                assert cohort_pl.select("person_id").is_unique().all()

        dataset_pl = self._read_data(dataset)

        # Hash patients
        if not pre_hashed:
            sequences = self.hash_patients(
                dataset=dataset_pl,
                attack_knowledge=self.attack_knowledge,
                pseudonymization_algorithm=self.pseudonymization_process["algorithm"],
            )
            dataset_pl = dataset_pl.join(sequences, on="person_id", how="inner")

        # Compute duration of stays
        dataset_pl = self.compute_difference_between_a_b(
            dataset_pl, "duration", "visit_end_date", "visit_start_date"
        )

        # Stays of patients to verify
        if (patients_to_check is None) and (cohort is not None):
            patients_to_check = (
                cohort_pl.sample(n=sample_size, seed=seed)
                .select("person_id")
                .to_series()
            )
        else:
            patients_to_check = pl.Series(patients_to_check)

        if batch_size is None:
            batch_size = len(patients_to_check)
        logger.debug(f"batch size : {batch_size}")

        # Process each chunk
        result_list = []
        if return_stays:
            patient_candidates_list = []
            stays_patient_candidates_list = []

        for i in range(0, len(patients_to_check), batch_size):
            chunk = patients_to_check[i : i + batch_size]

            logger.debug(f"chunk size : {len(chunk)}")

            (
                result_chunk,
                patient_candidates_chunk,
                stays_patient_candidates_chunk,
            ) = self.process_chunk(
                stays_pl=dataset_pl, patients_chunk=chunk, return_stays=return_stays
            )
            result_list.append(result_chunk)
            if return_stays:
                patient_candidates_list.append(patient_candidates_chunk)
                stays_patient_candidates_list.append(stays_patient_candidates_chunk)

        # concatenate all batchs
        result = pl.concat(result_list, rechunk=True, parallel=False)

        if return_stays:
            patient_candidates = pl.concat(
                patient_candidates_list, rechunk=True, parallel=False
            )
            stays_patient_candidates = pl.concat(
                stays_patient_candidates_list, rechunk=True, parallel=False
            )

            return result, patient_candidates, stays_patient_candidates
        else:
            return result
