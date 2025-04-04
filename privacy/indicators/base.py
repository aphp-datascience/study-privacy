from typing import Any, Dict

import pandas as pd


class ReliabilityIndicator:
    name = "UnnamedReliabilityIndicator"

    def __init__(self) -> None:
        pass

    def indicators_estimator(self, stays: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def compute(self, stays: pd.DataFrame, **kwargs) -> Any:
        raise NotImplementedError
