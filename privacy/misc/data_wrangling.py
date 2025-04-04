from typing import Any, List, Union

import numpy as np
import pandas as pd
import polars as pl
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame as sparkDataFrame


def make_sorted_list(x):
    l = list(x)
    l.sort()
    return l


def spark_filter_col_with_regex(
    df: sparkDataFrame,
    regex_patterns: Union[List[str], str],
    col: str,
) -> sparkDataFrame:
    """
    Filter column that match a regex pattern.

    Parameters
    ----------
    df: spark DataFrame
        the filter will be done over the column `col`
    col: str
        column name
    regex_patterns: List
        A list of regex patterns to match.
        All items of the list will be joined with the OR operator.

    Returns
    -------
    df_filtered: spark DataFrame with the rows that fulfill the conditions.

    """

    # Join all the terms wiht the OR operator
    if isinstance(regex_patterns, list):
        regex_patterns = [f"({i})" for i in regex_patterns]
        regex_patterns = r"|".join(regex_patterns)

    # Filter documents
    df_filtered = df.where(F.col(col).rlike(regex_patterns))

    return df_filtered


def pandas_to_polars_timedelta(td: pd.Timedelta) -> pl.Duration:
    return pl.duration(**dict(td.components._asdict()))


def flatten_list(nestedList) -> List[Any]:
    """"""
    # check if list is empty
    if not (bool(nestedList)):
        return nestedList

    # to check instance of list is empty or not
    if isinstance(nestedList[0], list):
        # call function with sublist as argument
        return flatten_list(*nestedList[:1]) + flatten_list(nestedList[1:])

    # call function with sublist as argument
    return nestedList[:1] + flatten_list(nestedList[1:])
