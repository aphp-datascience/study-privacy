import sys
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from privacy.attacks.access import p_access
from privacy.misc.utils import DataModule
from privacy.pipelines.reliability import pipe_reliability
from privacy.pipelines.uniqueness import pipe_uniqueness
from privacy.misc.constants import variations_conf_seasonal_epidemics

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

all_variations = list(variations_conf_table_1)
for item in variations_conf_seasonal_epidemics:
    if item not in all_variations:
        all_variations.append(item)

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
    specific_patients_to_check: Optional[List[str]] = None,
    scenario: str = "random_target",
    conf_name: str = "config_base"
):
    logger.info("Starting Table 1 computation")
    logger.info(f"Scenario: {scenario}")
    logger.info(f"Configuration: {conf_name}")
    
    conf_general_table_1 = dict(
        attack_knowledge=attack_knowledge,
        overall_cohort=None,
        overall_stays=None,
        cohort_to_check=None,
        stays_to_check=None,
        patients_to_check=None,
        seed=seed,
    )
    results_table_1 = []
    for cohort_name in cohorts_definitions.keys():
        logger.info(f"Cohort: {cohort_name}")
        logger.info(f"Attack scenario: {scenario}")
        logger.info(f"Attacker knowledge: {attack_knowledge}")

        stays_to_check = data.stays(cohort_name)
        cohort_to_check = data.cohort(cohort_name)
        if specific_patients_to_check is None:
            patients_to_check = data.patients_to_check(cohort_name, random_state=seed)
        else:
            patients_to_check = specific_patients_to_check
        indicators = cohorts_definitions[cohort_name]["indicators"]

        conf_general_table_1.update(
            dict(
                cohort=cohort_to_check,
                stays=stays_to_check,
                patients_to_check=patients_to_check,
            )
        )
        if scenario == "target_in_cohort":
            conf_general_table_1.update(
                dict(
                    overall_cohort=cohort_to_check,
                    overall_stays=stays_to_check,
                )
            )
        elif scenario == "random_target":
            conf_general_table_1.update(
                dict(
                    overall_cohort=data.cohort("all_population"),
                    overall_stays=data.stays("all_population"),
                )
            )

        for variation in all_variations:
            logger.info(f"Pseudonymisation algorithm: {variation.get('pseudonymization_algorithm')} - shift: {variation.get('high_general')}")
            conf_general_table_1.update(variation)

            uniqueness, remainder = pipe_uniqueness(**conf_general_table_1)

            reliability_indicator = pipe_reliability(
                indicators=indicators, **conf_general_table_1
            )

            if scenario == "target_in_cohort":
                access = 1.0
            else:
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
                    "remainder": remainder,
                }
            )

            results_table_1.append(variation.copy())

            if output_path is not None:
                table1_tmp = pd.DataFrame(results_table_1)
                table1_tmp.to_csv(output_path)

    table1 = pd.DataFrame(results_table_1)
    return table1
