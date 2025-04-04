import sys
from typing import List, Optional

import pandas as pd
from loguru import logger

from privacy import registry
from privacy.attacks.uniqueness import Uniqueness


def pipe_uniqueness(
    overall_cohort: pd.DataFrame,  # cohort total
    overall_stays: pd.DataFrame,  # all stays
    patients_to_check: pd.Series,  # patients de cohorte specifique
    pseudonymization_algorithm: str,  # one of {'BasePseudonymizer', 'BirthPseudonymizer', 'StayPseudonymizer'}
    attack_knowledge: List[str],
    return_tables: bool = False,
    seed: int = 55,
    batch_size: Optional[int] = None,
    **kwargs,
):
    # Apply pseudo
    pseudonimizer = registry.pseudonymizer.get(pseudonymization_algorithm)
    shifter = pseudonimizer(seed=seed, **kwargs)

    shifts = shifter.get_shifts_params()
    dataset = shifter.pseudonymize(stays=overall_stays, cohort=overall_cohort)
    pseudonymization_process = {
        "shifts": shifts,
        "algorithm": shifter.__class__.__name__,
    }

    # Check uniqueness
    u = Uniqueness(
        pseudonymization_process=pseudonymization_process,
        attack_knowledge=attack_knowledge,
        **kwargs,
    )

    result = u(
        dataset=dataset,
        patients_to_check=patients_to_check,
        batch_size=batch_size,
    )
    logger.debug(shifter.__class__.__name__)
    logger.debug(f"Attacker knowledge: {u.attack_knowledge}")

    counts = result.to_pandas().unique.value_counts(normalize=True)
    if True in counts.index:
        uniqueness = counts.loc[True]
    else:
        uniqueness = 0
    logger.info(f"Uniqueness {uniqueness}")

    if return_tables:
        return dict(
            result=result,
            # patient_candidates=patient_candidates,
            # stays_patient_candidates=stays_patient_candidates,
            stays_pseudo=dataset,
            pseudonymization_process=pseudonymization_process,
            uniqueness=uniqueness,
        )
    else:
        return uniqueness
