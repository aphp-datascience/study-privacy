from confit import Cli

from privacy import registry

app = Cli(pretty_exceptions_show_locals=False)
from typing import Any, List

import pandas as pd
from loguru import logger

from privacy.misc.utils import DataModule
from privacy.pipelines.table1 import get_table_1


@app.command(name="table1")
def main(
    cohorts_definitions: Any,
    attack_knowledge: List[str],
    output_path: str,
    scenario: str = "random_target",
    config_meta=None,
):
    conf_name =  config_meta["config_path"][0].stem
    data = DataModule(conf=config_meta["resolved_config"])

    table1 = get_table_1(
        cohorts_definitions=cohorts_definitions,
        data=data,
        attack_knowledge=attack_knowledge,
        seed=config_meta["resolved_config"]["general"]["seed"],
        output_path=None,
        scenario=scenario,
        conf_name=conf_name,
    )

    table1.to_csv(output_path)
    logger.info(f"Results saved at : {output_path}")


if __name__ == "__main__":
    app()
