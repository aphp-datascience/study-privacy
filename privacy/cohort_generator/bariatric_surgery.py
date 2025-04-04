from typing import List, Optional, Tuple

from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame as sparkDataFrame

from privacy.cohort_generator.icd10 import ICD10
from privacy.misc.utils import get_spark
from privacy.registry import registry

spark, sql = get_spark()


@registry.cohort_generator("BariatricSurgery")
class BariatricSurgery(ICD10):
    def __init__(
        self,
        db: str,
        start_date: str,
        end_date: str,
        procedure_codes: List[str],
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
            icd10_diagnosis=icd10_diagnosis,
            icd10_diagnosis_type=icd10_diagnosis_type,
            icd10_diagnosis_regex=icd10_diagnosis_regex,
            age_min_at_stay=age_min_at_stay,
        )

        self.procedure_codes = procedure_codes

    def lag_stay_tag(self, vo: sparkDataFrame) -> sparkDataFrame:
        # Add a visit_number
        window = Window.partitionBy("person_id").orderBy(
            [
                F.col("visit_start_date").asc(),
                F.col("visit_end_date").asc(),
            ]
        )
        vo = vo.withColumn(
            "previous_bariatric_surgery",
            F.lag("bariatric_surgery", default=False).over(window),
        )
        return vo

    def filter_by_procedure_codes(self, vo: sparkDataFrame) -> sparkDataFrame:
        po = sql("SELECT * FROM procedure_occurrence")
        po = po.select(
            [
                "visit_occurrence_id",
                "procedure_source_value",
            ]
        )

        po = po.filter(F.col("procedure_source_value").isin(self.procedure_codes))

        po = po.select("visit_occurrence_id").drop_duplicates(
            subset=["visit_occurrence_id"]
        )

        vo = vo.join(po, on="visit_occurrence_id", how="inner")

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

        # Filter by procedures codes
        vo = self.filter_by_procedure_codes(vo)

        # Filter by age
        vo = self.filter_by_age(vo)

        # Add tag to bariatric surgery
        vo = vo.withColumn("bariatric_surgery", F.lit(True))

        # Get cohort
        cohort = self.retrieve_cohort_info(vo)

        # Get all stays of cohort
        vo_all = self.retrieve_stays()
        vo_all = vo_all.join(cohort.select(["person_id"]), on="person_id", how="inner")

        # Add tag to all stays
        vo_all = vo_all.join(
            vo.select(["visit_occurrence_id", "bariatric_surgery"]),
            on="visit_occurrence_id",
            how="left",
        )

        vo_all = vo_all.fillna(False, subset=["bariatric_surgery"])

        # Propagate tag information
        vo_all = self.lag_stay_tag(vo_all)

        # Add hospitals
        vo_all = self.add_hospital_info_to_stays(vo_all)

        return vo_all, cohort
