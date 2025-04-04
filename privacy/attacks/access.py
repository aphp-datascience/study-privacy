from pandas import DataFrame


# Probability of access
def p_access(cohort: DataFrame, n_total: int) -> float:
    n_cohort = cohort["person_id"].nunique()
    p = n_cohort / n_total
    return p
