import concurrent
import itertools
import multiprocessing as mp
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from privacy.misc.utils import DataModule
from privacy.pipelines.uniqueness import pipe_uniqueness

logger.remove()
logger.add(sys.stderr, level="INFO")


class ConfGenerator:
    def __init__(
        self,
        list_shift_clinical: List[int],
        list_shift_birth: List[int],
        conf_general: Dict[str, Any],
    ):
        self.combinations = itertools.product(list_shift_clinical, list_shift_birth)
        self.conf_general = conf_general
        pass

    def __iter__(self):
        return self

    def __next__(self):
        shift_clinical, shift_birth = next(self.combinations)
        variation = dict(
            low_general=-shift_clinical,
            high_general=shift_clinical,
            low_birth_date=-shift_birth,
            high_birth_date=shift_birth,
        )

        self.conf_general.update(variation)
        return self.conf_general.copy()


def pipe_uniqueness_wrap(inputs):
    variation = {
        key: inputs.get(key)
        for key in ["low_general", "high_general", "low_birth_date", "high_birth_date"]
    }
    high_birth_date = variation["high_birth_date"]
    high_general = variation["high_general"]
    logger.info(f"shift general: {high_general} - shift birth {high_birth_date}")
    variation["uniqueness"] = pipe_uniqueness(**inputs)

    return variation


def get_table_simultaneous_variations_uniqueness(
    data: DataModule,
    attack_knowledge: List[str] = [
        "visit_start_date",
        "visit_end_date",
        "birth_date",
        "death_date",
        "gender",
        "hospital",
    ],
    list_shift_clinical: List[int] = [1, 2, 5, 10, 20, 50, 100, 200, 500, 700, 1000],
    list_shift_birth: List[int] = [1, 2, 5, 10, 20, 50, 100, 200, 500, 700, 1000],
    n: int = 50,
    seed: int = 55,
    batch_size: Optional[int] = 50,
    max_workers: int = 2,
):
    patients_to_check = data.patients_to_check("all_population", random_state=seed, n=n)
    conf_general = dict(
        overall_cohort=data.cohort("all_population"),
        overall_stays=data.stays("all_population"),
        seed=seed,
        pseudonymization_algorithm="BirthPseudonymizer",
        low_general=None,
        high_general=None,
        low_birth_date=None,
        high_birth_date=None,
        patients_to_check=patients_to_check,
        attack_knowledge=attack_knowledge,
        batch_size=batch_size,
    )
    results_table = []

    generator = ConfGenerator(
        list_shift_clinical=list_shift_clinical,
        list_shift_birth=list_shift_birth,
        conf_general=conf_general,
    )

    results_table = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        for result in executor.map(pipe_uniqueness_wrap, generator):
            results_table.append(result.copy())

        table = pd.DataFrame(results_table)
        return table
