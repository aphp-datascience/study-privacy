from typing import Any, Dict

import pandas as pd

from privacy.misc.constants import cohort_name_mapping
from privacy.misc.constants import pseudonimizer_name_mapping


def get_table_cohort_uniqueness(conf: Dict[str, Any]):
    path_table1 = conf["table1"]["output_path"]
    df = pd.read_csv(path_table1)
    df = df.query("~cohort_name.isin(['random'])").copy()

    df.cohort_name.replace(cohort_name_mapping, inplace=True)
    df["pseudonymization_algorithm"].replace(pseudonimizer_name_mapping, inplace=True)

    return df
