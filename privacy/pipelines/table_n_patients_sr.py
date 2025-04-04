from typing import Any, Dict

import pandas as pd
from privacy.misc.constants import pseudonimizer_name_mapping


def get_table_n_patients_sr(conf: Dict[str, Any]):
    path_table1 = conf["table1"]["output_path"]

    table1 = pd.read_csv(path_table1)

    columns = [
        "cohort_name",
        "n",
        "success_rate",
        "uniqueness",
        "access",
        "pseudonymization_algorithm",
        "low_general",
        "high_general",
    ]

    # other cohorts
    table1 = table1.query("~cohort_name.isin(['random'])").copy()
    table1.rename(columns={"n_cohort": "n"}, inplace=True)
    # cast
    table1["n"] = table1.n.astype(int)

    table_n_patients_sr = table1[columns]
    table_n_patients_sr["pseudonymization_algorithm"].replace(
        pseudonimizer_name_mapping, inplace=True
    )

    return table_n_patients_sr
