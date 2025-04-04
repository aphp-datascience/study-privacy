variations_conf_table_1 = (
    {
        "pseudonymization_algorithm": "NoPseudonymizer",
        "low_general": 0,
        "high_general": 0,
    },
    {
        "pseudonymization_algorithm": "BasePseudonymizer",
        "low_general": -10,
        "high_general": 10,
    },
    {
        "pseudonymization_algorithm": "BasePseudonymizer",
        "low_general": -100,
        "high_general": 100,
    },
    {
        "pseudonymization_algorithm": "BasePseudonymizer",
        "low_general": -1000,
        "high_general": 1000,
    },
    {
        "pseudonymization_algorithm": "BirthPseudonymizer",
        "low_general": -10,
        "high_general": 10,
        "low_birth_date": -10,
        "high_birth_date": 10,
    },
    {
        "pseudonymization_algorithm": "BirthPseudonymizer",
        "low_general": -100,
        "high_general": 100,
        "low_birth_date": -100,
        "high_birth_date": 100,
    },
    {
        "pseudonymization_algorithm": "BirthPseudonymizer",
        "low_general": -1000,
        "high_general": 1000,
        "low_birth_date": -1000,
        "high_birth_date": 1000,
    },
    {
        "pseudonymization_algorithm": "StayPseudonymizer",
        "low_general": -10,
        "high_general": 10,
    },
    {
        "pseudonymization_algorithm": "StayPseudonymizer",
        "low_general": -100,
        "high_general": 100,
    },
    {
        "pseudonymization_algorithm": "StayPseudonymizer",
        "low_general": -1000,
        "high_general": 1000,
    },
)

import sys
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from privacy.attacks.access import p_access
from privacy.misc.utils import DataModule
from privacy.pipelines.reliability import pipe_reliability
from privacy.pipelines.uniqueness import pipe_uniqueness

logger.remove()
logger.add(sys.stderr, level="INFO")


def get_table_1(
    cohorts_definitions: Dict[str, Any],
    data: DataModule,
    attack_knowledge: List[str] = [
        "visit_start_date",
        "visit_end_date",
        "birth_date",
        "death_date",
        "gender",
        "hospital",
    ],
    seed: int = 55,
    output_path: Optional[str] = None,
):
    conf_general_table_1 = dict(
        attack_knowledge=attack_knowledge,
        overall_cohort=data.cohort("all_population"),
        overall_stays=data.stays("all_population"),
        cohort_to_check=None,
        stays_to_check=None,
        patients_to_check=None,
        seed=seed,
    )
    results_table_1 = []
    for cohort_name in cohorts_definitions.keys():
        logger.info(f"Cohort: {cohort_name}")
        stays_to_check = data.stays(cohort_name)
        cohort_to_check = data.cohort(cohort_name)
        patients_to_check = data.patients_to_check(cohort_name, random_state=seed)
        indicators = cohorts_definitions[cohort_name]["indicators"]

        conf_general_table_1.update(
            dict(
                cohort=cohort_to_check,
                stays=stays_to_check,
                patients_to_check=patients_to_check,
            )
        )

        for variation in variations_conf_table_1:
            conf_general_table_1.update(variation)

            uniqueness = pipe_uniqueness(**conf_general_table_1)

            reliability_indicator = pipe_reliability(
                indicators=indicators, **conf_general_table_1
            )

            access = p_access(
                cohort=cohort_to_check,
                n_total=data.all_population_cohort.person_id.nunique(),
            )
            success_rate = access * uniqueness
            n_cohort = cohort_to_check["person_id"].nunique()
            variation.update(
                {
                    "cohort_name": cohort_name,
                    "uniqueness": uniqueness,
                    "reliability_indicator": reliability_indicator,
                    "access": access,
                    "success_rate": success_rate,
                    "n_cohort": n_cohort,
                }
            )

            results_table_1.append(variation.copy())

            if output_path is not None:
                table1_tmp = pd.DataFrame(results_table_1)
                table1_tmp.to_csv(output_path)

    table1 = pd.DataFrame(results_table_1)
    return table1
