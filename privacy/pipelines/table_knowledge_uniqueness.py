import concurrent
import sys
from typing import List, Optional, Tuple

import pandas as pd
from loguru import logger

from privacy.attacks.access import p_access
from privacy.misc.constants import attack_knowledge_name_mapping
from privacy.misc.utils import DataModule
from privacy.pipelines.table1 import variations_conf_table_1
from privacy.pipelines.uniqueness import pipe_uniqueness

logger.remove()
logger.add(sys.stderr, level="DEBUG")


def get_table_knowledge_uniqueness(
    data: DataModule,
    attack_knowledges: Tuple[List[str]] = (
        [
            "visit_start_date",
            "visit_end_date",
            "birth_date",
            "death_date",
            "gender",
            "hospital",
        ],
        [
            "visit_start_date",
            "visit_end_date",
            "birth_date",
            "death_date",
            "gender",
        ],
        [
            "visit_start_date",
            "birth_date",
            "death_date",
            "gender",
        ],
        [
            "birth_date",
            "death_date",
            "gender",
        ],
    ),
    cohort_name: str = "all_population",
    seed: int = 55,
    batch_size: int = 40,
    output_path: Optional[str] = None,
):
    conf_general = dict(
        overall_cohort=data.cohort("all_population"),
        overall_stays=data.stays("all_population"),
        cohort_to_check=None,
        stays_to_check=None,
        patients_to_check=None,
        seed=seed,
        batch_size=batch_size,
    )
    results_table = []

    patients_to_check = data.patients_to_check(
        cohort_name,
        random_state=seed,
    )

    conf_general.update(
        dict(
            patients_to_check=patients_to_check,
        )
    )

    for variation in variations_conf_table_1:
        conf_general.update(variation)

        for attack_knowledge in attack_knowledges:
            conf_general.update(dict(attack_knowledge=attack_knowledge))

            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                uniqueness = executor.submit(pipe_uniqueness, **conf_general).result()

            attack_knowledge_str = " + ".join(attack_knowledge)
            variation.update(
                {
                    "cohort_name": cohort_name,
                    "uniqueness": uniqueness,
                    "attack_knowledge": attack_knowledge_str,
                }
            )

            results_table.append(variation.copy())

            if output_path is not None:
                table_tmp = pd.DataFrame(results_table)
                table_tmp.attack_knowledge.replace(
                    attack_knowledge_name_mapping, inplace=True
                )

                table_tmp.to_csv(output_path)

    table = pd.DataFrame(results_table)
    table.attack_knowledge.replace(attack_knowledge_name_mapping, inplace=True)

    return table
