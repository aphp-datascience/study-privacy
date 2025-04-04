from typing import List, Optional, Tuple

from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame as sparkDataFrame

from privacy.cohort_generator.base import BaseCohortGenerator
from privacy.misc.data_wrangling import spark_filter_col_with_regex
from privacy.misc.utils import get_spark
from privacy.registry import registry

spark, sql = get_spark()


@registry.cohort_generator("ICD10")
class ICD10(BaseCohortGenerator):
    def __init__(
        self,
        db: str,
        start_date: str,
        end_date: str,
        icd10_diagnosis: Optional[List[str]] = None,
        icd10_diagnosis_type: Optional[List[str]] = None,
        icd10_diagnosis_regex: Optional[List[str]] = None,
        visit_type: List[str] = ["hospitalisés"],
        age_min_at_stay: Optional[int] = 0,
    ) -> None:
        super().__init__(
            db=db,
            start_date=start_date,
            end_date=end_date,
            visit_type=visit_type,
            age_min_at_stay=age_min_at_stay,
        )

        self.icd10_diagnosis = icd10_diagnosis
        self.icd10_diagnosis_type = icd10_diagnosis_type
        self.icd10_diagnosis_regex = icd10_diagnosis_regex

    def filter_stays_by_icd_10_diagnosis(self, vo: sparkDataFrame) -> sparkDataFrame:
        # Filter by icd10 diagnosis
        co = sql("SELECT *  FROM condition_occurrence")
        co = co.select(
            [
                "visit_occurrence_id",
                "condition_source_value",
                "condition_status_source_value",
                "condition_occurrence_id",
                "cdm_source",
            ]
        )

        co_collected = spark.createDataFrame([], schema=co.schema)
        if self.icd10_diagnosis is not None:
            co_exact = co.filter(
                F.col("condition_source_value").isin(self.icd10_diagnosis)
            )
            co_collected = co_collected.union(co_exact)

        if self.icd10_diagnosis_regex is not None:
            co_regex = spark_filter_col_with_regex(
                co, self.icd10_diagnosis_regex, "condition_source_value"
            )
            co_collected = co_collected.union(co_regex)

        co_collected = co_collected.drop_duplicates(subset=["condition_occurrence_id"])

        if self.icd10_diagnosis_type is not None:
            co_collected = co_collected.filter(
                F.col("condition_status_source_value").isin(self.icd10_diagnosis_type)
            )

        co_collected = co_collected.select("visit_occurrence_id").drop_duplicates(
            subset=["visit_occurrence_id"]
        )

        vo = vo.join(co_collected, on="visit_occurrence_id", how="inner")

        vo.cache()
        return vo

    def __call__(
        self,
    ) -> Tuple[sparkDataFrame, sparkDataFrame]:
        # Set DB
        sql(f"use {self.db}")

        # Get stays
        vo = self.retrieve_stays()

        # Filter opposed patients
        vo = self.filter_opposed_patients(vo)

        # Filter by icd10
        vo = self.filter_stays_by_icd_10_diagnosis(vo)

        # Filter by age
        vo = self.filter_by_age(vo)

        # Get cohort
        cohort = self.retrieve_cohort_info(vo)

        # Add hospitals
        vo = self.add_hospital_info_to_stays(vo)

        return vo, cohort
