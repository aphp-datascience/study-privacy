from typing import List, Optional, Tuple

from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame as sparkDataFrame

from privacy.misc.constants import mapping_code_hospital_short_name
from privacy.misc.utils import get_spark

spark, sql = get_spark()


class BaseCohortGenerator:
    def __init__(
        self,
        db: str,
        start_date: str,
        end_date: str,
        visit_type: Optional[List[str]] = ["hospitalisés"],
        age_min_at_stay: Optional[int] = 0,
    ) -> None:
        self.db = db
        self.start_date = start_date
        self.end_date = end_date
        self.visit_type = visit_type
        self.age_min_at_stay = age_min_at_stay

    def retrieve_stays(self) -> sparkDataFrame:
        vo = sql("SELECT * FROM visit_occurrence")
        vo = vo.select(
            [
                "person_id",
                "visit_occurrence_id",
                "visit_start_date",
                "visit_end_date",
                "care_site_id",
                "visit_source_value",
            ]
        )

        # Filter by type of visit and dates
        if self.visit_type is not None:
            vo = vo.filter(F.col("visit_source_value").isin(self.visit_type))
        vo = vo.filter(F.col("visit_start_date") >= self.start_date)
        vo = vo.filter(F.col("visit_start_date") <= self.end_date)

        # # Add a visit_number
        window = Window.partitionBy("person_id").orderBy(
            [
                F.col("visit_start_date").asc(),
                F.col("visit_end_date").asc(),
            ]
        )
        vo = vo.withColumn("visit_number", F.row_number().over(window))

        vo.cache()

        return vo

    def retrieve_cohort_info(self, vo: sparkDataFrame) -> sparkDataFrame:
        # Get cohort
        cohort = vo.select("person_id").drop_duplicates(subset=["person_id"])

        # Add person details
        person = sql(
            "SELECT person_id, birth_datetime, death_datetime, gender_source_value from person"
        )
        person = person.withColumn("birth_date", F.to_date(F.col("birth_datetime")))
        person = person.withColumn("death_date", F.to_date(F.col("death_datetime")))
        person = person.drop("birth_datetime", "death_datetime")

        cohort_w_info = person.join(cohort, on="person_id")

        cohort_w_info = cohort_w_info.drop_duplicates(subset=["person_id"])

        return cohort_w_info

    def add_hospital_info_to_stays(self, vo: sparkDataFrame) -> sparkDataFrame:
        # Add hospitals
        cs = sql(
            "SELECT care_site_id, care_site_source_value from care_site where care_site_type_source_value=='Hôpital'"
        )

        mapping_code_hospital_short_name_spark = spark.createDataFrame(
            mapping_code_hospital_short_name
        )
        cs = cs.join(
            mapping_code_hospital_short_name_spark,
            on="care_site_source_value",
            how="left",
        )
        cs = cs.drop("care_site_source_value")

        vo = vo.join(cs, on="care_site_id", how="left")

        vo = vo.fillna(value="Unknown", subset=["care_site_short_name"])
        return vo

    def filter_by_age(self, vo: sparkDataFrame) -> sparkDataFrame:
        if self.age_min_at_stay is not None:
            person = sql("SELECT person_id, birth_datetime from person")
            person = person.withColumn("birth_date", F.to_date(F.col("birth_datetime")))
            person = person.drop("birth_datetime")
            vo = vo.join(person, on="person_id")

            vo = vo.withColumn(
                "age_days", F.datediff(F.col("visit_start_date"), F.col("birth_date"))
            )
            vo = vo.filter(F.col("age_days") >= (self.age_min_at_stay * 365))
            vo = vo.drop("birth_date", "age_days")
        return vo

    def filter_opposed_patients(self, df: sparkDataFrame):
        person = sql(
            "SELECT person_id from person WHERE status_source_value == 'Actif'"
        )

        df = df.join(person, on="person_id", how="inner")
        return df

    def __call__(
        self,
    ) -> Tuple[sparkDataFrame, sparkDataFrame]:
        # Set DB
        sql(f"use {self.db}")

        # Get stays
        vo = self.retrieve_stays()

        # Filter by age
        vo = self.filter_by_age(vo)

        # Get cohort
        cohort = self.retrieve_cohort_info(vo)

        # Keep only patients in cohort
        vo = vo.join(cohort.select("person_id"), on="person_id", how="inner")

        # Add hospitals
        vo = self.add_hospital_info_to_stays(vo)

        # Filter opposed patients
        vo = self.filter_opposed_patients(vo)

        return vo, cohort
