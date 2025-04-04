from typing import List

import pandas as pd
from loguru import logger

from privacy import registry
from privacy.indicators.base import ReliabilityIndicator


def get_pseudonimized_dataset(
    cohort: pd.DataFrame,
    stays: pd.DataFrame,
    pseudonymization_algorithm: str,  # one of {'BasePseudonymizer', 'BirthPseudonymizer', 'StayPseudonymizer'},
    seed: int = 55,
    **kwargs,
):
    # Apply pseudo
    pseudonimizer = registry.pseudonymizer.get(pseudonymization_algorithm)
    shifter = pseudonimizer(seed=seed, **kwargs)
    dataset = shifter.pseudonymize(stays, cohort, **kwargs)
    return dataset


def pipe_reliability(
    cohort: pd.DataFrame,
    stays: pd.DataFrame,
    indicators: List[
        ReliabilityIndicator
    ],  # one of {'Age', 'KLDivergence', 'Readmission'}
    pseudonymization_algorithm: str,  # one of {'NoPseudonymizer','BasePseudonymizer', 'BirthPseudonymizer', 'StayPseudonymizer'},
    seed: int = 55,
    **kwargs,
):
    # Apply pseudo
    dataset = get_pseudonimized_dataset(
        cohort=cohort,
        stays=stays,
        pseudonymization_algorithm=pseudonymization_algorithm,
        seed=seed,
        **kwargs,
    )

    results = {}
    kwargs_reliability = {"pseudonymization_algorithm": pseudonymization_algorithm}
    for indicator in indicators:
        results.update(indicator.indicators_estimator(dataset, **kwargs_reliability))

    logger.info(f"Reliability {results}")

    reliability_indicator = sum(results.values())
    if len(indicators) > 0:
        normalized_reliability_indicator = reliability_indicator / len(indicators)
        return normalized_reliability_indicator
    else:
        return reliability_indicator
