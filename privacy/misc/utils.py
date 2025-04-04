import os
from typing import Any, Dict, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import parquet
from pyspark.sql import SparkSession


class arrowConnector:
    def __init__(self, path_table=None, db=None, table=None):
        self.path_table = path_table
        self.db = db
        if db and table:
            self.path_table = (
                sql(f"desc formatted {db}.{table}")  # noqa: F821 # type: ignore
                .filter("col_name=='Location'")
                .collect()[0]
                .data_type
            )
            self.db = os.path.dirname(self.path_table)

        if path_table:
            self.path_table = path_table

        if (db) and (table is None):
            self.db = db

    def get_pd_fragment(
        self,
        path_table=None,
        table_name=None,
        types_mapper=None,
        integer_object_nulls=True,
        date_as_object=False,
    ):
        if path_table:
            self.path_table = path_table

        if table_name:
            self.path_table = os.path.join(self.db, table_name)

        # Import the parquet as ParquetDataset
        parquet_ds = pa.parquet.ParquetDataset(
            self.path_table, use_legacy_dataset=False
        )

        # Partitions of ds
        fragments = iter(parquet_ds.fragments)

        # Set initial length
        length = 0

        # One partition
        while length < 1:
            fragment = next(fragments)

            # pyarrow.table of a fragment
            table = fragment.to_table()

            length = len(table)

        # Import to pandas the fragment
        table_pd = table.to_pandas(
            types_mapper=types_mapper,
            integer_object_nulls=integer_object_nulls,
            date_as_object=date_as_object,
        )
        return table_pd

    def count_fragments_length(self, path_table=None, table_name=None):
        if path_table:
            self.path_table = path_table

        if table_name:
            self.path_table = os.path.join(self.db, table_name)
        # Import the parquet as ParquetDataset
        parquet_ds = pa.parquet.ParquetDataset(
            self.path_table, use_legacy_dataset=False
        )

        # Partitions of ds
        fragments = iter(parquet_ds.fragments)

        lengths = []
        for fragment in fragments:
            # pyarrow.table of a fragment
            table = fragment.to_table()
            lengths.append(len(table))

        return lengths

    def get_pd_table(
        self,
        path_table=None,
        table_name=None,
        types_mapper=None,
        integer_object_nulls=True,
        date_as_object=False,
        filter_values_keep=None,
        filter_values_avoid=None,
        cast_to_tz: Optional[str] = None,
        filter_col="person_id",
    ):
        if path_table:
            self.path_table = path_table

        if table_name:
            self.path_table = os.path.join(self.db, table_name)

        table = pa.parquet.read_table(self.path_table)
        if filter_values_keep:
            table = table.filter(pc.field(filter_col).isin(filter_values_keep))
        if filter_values_avoid:
            table = table.filter(
                pc.bit_wise_not(pc.field(filter_col).isin(filter_values_avoid))
            )

        df = table.to_pandas(
            date_as_object=date_as_object,
            types_mapper=types_mapper,
            integer_object_nulls=integer_object_nulls,
        )

        if cast_to_tz is not None:
            df = self.cast_to_tz(df, tz=cast_to_tz)
        return df

    @staticmethod
    def cast_to_tz(df, tz="Europe/Paris"):
        cols = df.select_dtypes(include=["datetime64"]).columns
        for col in cols:
            df[col] = df[col].dt.tz_localize("UTC")

            df[col] = df[col].dt.tz_convert(tz)
        return df


# Create Spark Session
def get_spark():
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    sql = spark.sql

    return spark, sql


class DataModule:
    def __init__(self, conf: Dict[str, Any]) -> None:
        self.conf = conf
        self.seed = conf["general"]["seed"]
        self._cache()
        self.n_patients_to_check = conf["general"]["n_patients_to_check"]

    @staticmethod
    def sample(dfcohort: pd.DataFrame, dfstays: pd.DataFrame, n_1000: int, seed: int):
        n = n_1000 * 1000

        cohort_sample = dfcohort.sample(n=n, random_state=seed)

        stays_sample = dfstays.merge(
            cohort_sample[["person_id"]],
            on="person_id",
            how="inner",
            validate="many_to_one",
        )
        return cohort_sample, stays_sample

    def _cache(self):
        for c in self.conf["cohorts"]["cohorts_definitions"]:
            path_save_stays = self.conf["cohorts"]["cohorts_definitions"][c][
                "path_save_stays"
            ]

            path_save_cohort = self.conf["cohorts"]["cohorts_definitions"][c][
                "path_save_cohort"
            ]
            dfcohort = pd.read_parquet(path_save_cohort)

            dfstays = pd.read_parquet(path_save_stays)

            if c == "random":
                n_1000 = self.conf["cohorts"]["cohorts_definitions"][c][
                    "cohort_selector"
                ]["n_1000"]

                dfcohort, dfstays = self.sample(dfcohort, dfstays, n_1000, self.seed)

            setattr(self, f"{c}_cohort", dfcohort)
            setattr(self, f"{c}_stays", dfstays)

    def stays(self, cohort_name: str):
        return getattr(self, cohort_name + "_stays")

    def cohort(self, cohort_name: str):
        return getattr(self, cohort_name + "_cohort")

    def patients_to_check(
        self,
        cohort_name: str,
        n: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        if n is None:
            n = self.n_patients_to_check
        if random_state is None:
            random_state = self.seed
        cohort = getattr(self, cohort_name + "_cohort")
        patients_to_check = cohort.sample(n, random_state=random_state).person_id
        return patients_to_check


def get_dir_path(file):
    path_conf_file = os.path.dirname(os.path.realpath(file))
    return path_conf_file


def build_path(file, relative_path):
    """
    Function to build an absolut path.

    Parameters
    ----------
    file: main file from where we are calling. It could be __file__
    relative_path: str,
        relative path from the main file to the desired output

    Returns
    -------
    path: absolute path
    """
    dir_path = get_dir_path(file)
    path = os.path.abspath(os.path.join(dir_path, relative_path))
    path_dir = os.path.dirname(path)
    if not os.path.isdir(path_dir):
        os.makedirs(path_dir)

    return path
