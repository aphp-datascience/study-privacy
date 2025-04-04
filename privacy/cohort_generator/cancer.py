from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame as sparkDataFrame
from pyspark.sql.types import StringType
from pyspark.sql.window import Window

from privacy.cohort_generator.base import BaseCohortGenerator
from privacy.cohort_generator.icd10 import ICD10
from privacy.misc.data_wrangling import flatten_list, spark_filter_col_with_regex
from privacy.misc.utils import get_spark
from privacy.registry import registry

spark, sql = get_spark()


@registry.cohort_generator("Cancer")
class Cancer(BaseCohortGenerator):
    icd10_all_cancer = {
        "anus": {"icd10_code": ["C21", "D013"]},
        "biliary_duct": {"icd10_code": ["C221", "C23", "C24", "D015", "D376"]},
        "bladder": {
            "icd10_code": [
                "C66",
                "C67",
                "C68",
                "D090",
                "D091",
                "D412",
                "D413",
                "D414",
                "D417",
                "D419",
            ]
        },
        "bowel": {"icd10_code": ["C17", "D014", "D372"]},
        "breast": {"icd10_code": ["C50", "D05", "D486"]},
        "cervix": {"icd10_code": ["C53", "D06"]},
        "CNS": {
            "icd10_code": [
                "C70",
                "C71",
                "C720",
                "C722",
                "C723",
                "C728",
                "C729",
                "D42",
                "D430",
                "D431",
                "D432",
                "D434",
                "D437",
                "D439",
            ]
        },
        "colon": {
            "icd10_code": [
                "C18",
                "C19",
                "D010",
                "D011",
                "D374",
                "D373",
                "C20",
                "D012",
                "D375",
            ]
        },  # colon + rectum
        "CUP": {
            "icd10_code": [
                "C76",
                "C80",
                "C97",
                "D097",
                "D099",
                "D483",
                "D487",
                "D489",
            ]
        },
        "endometrium": {"icd10_code": ["C54", "C55", "D070", "D390"]},
        "eye": {"icd10_code": ["C69", "D092"]},
        "gastric": {"icd10_code": ["C16", "D002", "D371"]},
        "head_neck": {
            "icd10_code": [
                "C0",
                "C10",
                "C11",
                "C12",
                "C13",
                "C14",
                "C30",
                "C31",
                "C32",
                "D000",
                "D020",
                "D370",
                "D380",
            ]
        },
        "hodgkin_lymphoma": {"icd10_code": ["C81"]},
        "kidney": {"icd10_code": ["C64", "C65", "D410", "D411"]},
        "leukemia": {
            "icd10_code": [
                "C91",
                "C92",
                "C93",
                "C940",
                "C941",
                "C942",
                "C943",
                "C944",
                "C945",
                "C947",
                "C95",
            ]
        },
        "liver": {"icd10_code": ["C220", "C222", "C223", "C224", "C227", "C229"]},
        "lung": {"icd10_code": ["C33", "C34", "D021", "D022"]},
        "mesothelioma": {"icd10_code": ["C45"]},
        "myeloma": {"icd10_code": ["C90"]},
        "nonhodgkin_lymphoma": {"icd10_code": ["C82", "C83", "C84", "C85", "C86"]},
        "oesophagus": {"icd10_code": ["C15", "D001"]},
        "osteosarcoma": {"icd10_code": ["C40", "C41", "D480"]},
        "other_digestive": {
            "icd10_code": ["C26", "C48", "D017", "D019", "D377", "D379", "D484"]
        },
        "other_endocrinial": {
            "icd10_code": [
                "C74",
                "C75",
                "D093",
                "D441",
                "D442",
                "D443",
                "D444",
                "D445",
                "D446",
                "D447",
                "D448",
                "D449",
            ]
        },
        "other_gynecology": {
            "icd10_code": [
                "C51",
                "C52",
                "C57",
                "C58",
                "D071",
                "D072",
                "D073",
                "D392",
                "D397",
                "D399",
            ]
        },
        "other_hematologic_malignancies": {
            "icd10_code": ["C46", "C88", "C96", "C946", "D45", "D46", "D47"]
        },
        "other_pneumology": {
            "icd10_code": [
                "C37",
                "C38",
                "C39",
                "D023",
                "D024",
                "D382D383",
                "D384",
                "D385",
                "D386",
            ]
        },
        "other_skin": {"icd10_code": ["C44", "D04", "D485"]},
        "other_urothelial": {
            "icd10_code": ["C60", "C63", "D074", "D076", "D407", "D409"]
        },
        "ovary": {"icd10_code": ["C56", "D391"]},
        "pancreas": {
            "icd10_code": [
                "C250",
                "C251",
                "C252",
                "C253",
                "C255",
                "C256",
                "C257",
                "C258",
                "C259",
            ]
        },
        "pancreas_endocrine": {"icd10_code": ["C254"]},
        "PNS": {"icd10_code": ["C47", "C721", "C724", "C725", "D433", "D482"]},
        "prostate": {"icd10_code": ["C61", "D075", "D400"]},
        "soft_tissue": {"icd10_code": ["C49", "D481"]},
        "skin": {"icd10_code": ["C43", "D03"]},
        "testis": {"icd10_code": ["C62", "D401"]},
        "thyroid": {"icd10_code": ["C73", "D440"]},
    }

    def __init__(
        self,
        db: str,
        start_date: str,
        end_date: str,
        cancer_types: List[str],
        visit_type: Optional[List[str]] = None,
        age_min_at_stay: Optional[int] = 0,
        icd10_all_cancer_update: Dict[str, Dict[str, List[str]]] = None,
        clean_period_years: int = 2,
        double_cancer_exclusion: bool = True,
        claim_data_type: str = "ORBIS",
        diagnostic_types: Optional[Union[str, List[str]]] = ["DP", "DR"],
        treatments: Dict[str, Dict[str, List[str]]] = dict(
            surgery=dict(
                procedure=[
                    "HGFA014",
                    "HNFA001",
                    "HNFA002",
                    "HNFA004",
                    "HNFA005",
                    "HNFA006",
                    "HNFA007",
                    "HNFA008",
                    "HNFA010",
                    "HNFA011",
                    "HNFA013",
                    "HNFC001",
                    "HNFC002",
                    "HNFC028",
                ],
                priority=1,
            ),
            anticancer_therapy=dict(condition=["Z511"], priority=2),
            best_supportive_care=dict(condition=["Z515"], priority=3),
        ),
        debug: bool = False,
    ) -> None:
        super().__init__(db, start_date, end_date, visit_type, age_min_at_stay)

        if icd10_all_cancer_update is not None:
            self.icd10_all_cancer.update(icd10_all_cancer_update)

        self.cancer_types = cancer_types
        self.clean_period_years = clean_period_years
        self.double_cancer_exclusion = double_cancer_exclusion
        self.claim_data_type = claim_data_type
        self.diagnostic_types = diagnostic_types
        self.debug = debug
        self.treatments = treatments

    def get_co(
        self,
        col_names=[
            "person_id",
            "visit_occurrence_id",
            "condition_source_value",
            "condition_status_source_value",
            "condition_occurrence_id",
            "cdm_source",
            "condition_start_datetime",
        ],
    ):
        sql(f"use {self.db}")
        co = sql("SELECT *  FROM condition_occurrence")

        co = co.select(col_names)

        # Make columns with the ICD10 code with 2 and 3 digits. Example: C40 & C401
        co = co.withColumn(
            "condition_source_value_short_2",
            F.substring("condition_source_value", 1, 3),
        )
        co = co.withColumn(
            "condition_source_value_short_3",
            F.substring("condition_source_value", 1, 4),
        )
        return co

    def get_po(
        self,
        col_names=[
            "person_id",
            "visit_occurrence_id",
            "procedure_source_value",
            "procedure_occurrence_id",
            "cdm_source",
            "procedure_datetime",
        ],
    ):
        sql(f"use {self.db}")
        po = sql("SELECT *  FROM procedure_occurrence")

        po = po.select(col_names)
        return po

    def get_cohort(
        self,
        **kwargs: Any,
    ) -> sparkDataFrame:
        # all diag
        co = self.get_co()
        # co = co.join(patients0, on="person_id", how="inner")

        if self.diagnostic_types is not None:
            co = co.filter(
                F.col("condition_status_source_value").isin(self.diagnostic_types)
            )

        # Classify each ICD10 code into a family
        cancer_icd10_code_dict = {
            key: value["icd10_code"] for key, value in self.icd10_all_cancer.items()
        }
        codes_cancer_pd = pd.DataFrame(
            cancer_icd10_code_dict.items(), columns=["cancer", "code"]
        )
        codes_cancer_pd = codes_cancer_pd.explode("code", ignore_index=True)
        codes_cancer = spark.createDataFrame(codes_cancer_pd)

        ## We keep only lines related to cancer & add classification
        icd10_patients1 = co.join(
            codes_cancer.hint("broadcast"),
            on=(
                (co.condition_source_value_short_2 == codes_cancer.code)
                | (co.condition_source_value_short_3 == codes_cancer.code)
            ),
            how="inner",
        )

        # Count number of different families of ICD10 values for each patient
        # and detect patients only with one familly of codes (monocancer)
        windowSpec = Window.partitionBy("person_id")
        icd10_patients1 = icd10_patients1.withColumn(
            "n_distinct_cancer",
            F.size(F.collect_set("cancer").over(windowSpec)),
        )
        if self.double_cancer_exclusion:
            icd10_patients2 = icd10_patients1.where(F.col("n_distinct_cancer") == 1)
        else:
            icd10_patients2 = icd10_patients1

        # Time delta between two consecutive ICD10 codes for a patient & cancer
        windowSpec = Window.partitionBy(["person_id", "cancer"]).orderBy(
            "condition_start_datetime"
        )
        icd10_patients2 = icd10_patients2.withColumn(
            "condition_start_datetime_1",
            F.lag("condition_start_datetime").over(windowSpec),
        )
        icd10_patients2 = icd10_patients2.withColumn(
            "delta_code_date",
            F.datediff("condition_start_datetime", "condition_start_datetime_1"),
        )

        # # Select only new patients
        clean_period_days = self.clean_period_years * 365

        windowSpec = Window.partitionBy(["person_id", "cancer"])
        icd10_patients2 = icd10_patients2.withColumn(
            "new_case",
            F.when(
                F.col("delta_code_date") <= clean_period_days,
                False,
            ).otherwise(True),
        )

        # Select only cancer of interest
        icd10_patients3 = icd10_patients2.filter(
            F.col("cancer").isin(self.cancer_types)
        )

        # Select only new cases
        icd10_patients4 = icd10_patients3.filter(F.col("new_case"))
        icd10_patients4 = icd10_patients4.drop("new_case")

        # coded between dates
        icd10_patients5 = icd10_patients4.filter(
            (F.col("condition_start_datetime") >= self.start_date)
            & (F.col("condition_start_datetime") < self.end_date)
        )

        # Filter opposed patients
        patients_cancer = self.filter_opposed_patients(icd10_patients5)

        # Aggregate by code
        patients_cancer = patients_cancer.groupby(["person_id", "cancer"]).agg(
            F.to_date(F.min(F.col("condition_start_datetime"))).alias(
                "first_diagnostic_code"
            )
        )

        # Add birth_date and death_date
        cohort_info = self.retrieve_cohort_info(patients_cancer)

        # Keep only patients in cohort
        patients_cancer = patients_cancer.join(cohort_info, on="person_id", how="inner")

        return patients_cancer

    def _get_treatment_spark_df(self, source):
        df_treatments = pd.DataFrame.from_dict(
            self.treatments,
            orient="index",
        )
        df_treatments.index.rename("treatment", inplace=True)
        df_treatments.reset_index(inplace=True)
        conditions_procedures = df_treatments[
            ["treatment", "priority", source]
        ].explode(column=source)
        conditions_procedures.dropna(subset=[source], inplace=True)
        conditions_procedures_spark = spark.createDataFrame(conditions_procedures)
        return conditions_procedures_spark

    def get_treatments(self, cohort):
        procedures = self._get_treatment_spark_df("procedure")
        conditions = self._get_treatment_spark_df("condition")
        cohort = cohort.select("person_id").drop_duplicates()
        # Get co
        co = self.get_co()
        co = co.join(cohort, on="person_id")
        po = self.get_po()
        po = po.join(cohort, on="person_id")

        ##
        cols = [
            "person_id",
            "visit_occurrence_id",
            "treatment",
            "treatment_start_datetime",
            "priority",
        ]
        treatments1 = co.join(
            conditions.hint("broadcast"),
            on=(
                (co.condition_source_value_short_2 == conditions.condition)
                | (co.condition_source_value_short_3 == conditions.condition)
            ),
            how="inner",
        )
        treatments1 = treatments1.withColumnRenamed(
            "condition_start_datetime", "treatment_start_datetime"
        ).select(cols)

        treatments2 = po.join(
            procedures.hint("broadcast"),
            on=(po.procedure_source_value == procedures.procedure),
            how="inner",
        )
        treatments2 = treatments2.withColumnRenamed(
            "procedure_datetime", "treatment_start_datetime"
        ).select(cols)

        treatments = treatments1.union(treatments2)

        treatments = treatments.filter(
            (F.col("treatment_start_datetime") >= self.start_date)
            & (F.col("treatment_start_datetime") < self.end_date)
        )

        return treatments

    def __call__(self) -> Tuple[sparkDataFrame, sparkDataFrame]:
        cohort = self.get_cohort()
        cohort.cache()
        treatments = self.get_treatments(cohort)
        treatments = treatments.withColumn(
            "visit_start_date", F.to_date("treatment_start_datetime")
        )
        treatments = treatments.drop("treatment_start_datetime")
        treatments = treatments.drop_duplicates(
            subset=[
                "person_id",
                "visit_occurrence_id",
                "treatment",
                "visit_start_date",
            ]
        )
        stays = self.retrieve_stays()
        stays = stays.drop("person_id", "visit_start_date", "visit_number")

        # >> each stay equals to each come to the hospital (anticancer therapy)
        stays_treatments = treatments.join(stays, how="left", on="visit_occurrence_id")

        stays_treatments = self.add_hospital_info_to_stays(stays_treatments)

        window = Window.partitionBy("person_id").orderBy(
            [
                F.col("visit_start_date").asc(),
                F.col("visit_end_date").asc(),
                F.col("priority").asc(),
            ]
        )
        stays_treatments = stays_treatments.withColumn(
            "visit_number", F.row_number().over(window)
        )

        return stays_treatments, cohort
