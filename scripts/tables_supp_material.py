from typing import Dict, List

from confit import Cli

from privacy import registry

app = Cli(pretty_exceptions_show_locals=False)

from loguru import logger

from privacy.misc.utils import DataModule
from privacy.pipelines.tables_supp_material import SuppMaterialTablesGenerator


@app.command(name="tables_supp_material")
def main(
    bins_stay_length: List[int],
    n_test_uniqueness: int,
    random_state: int,
    output_paths: Dict[str, str],
    config_meta=None,
):
    data = DataModule(conf=config_meta["resolved_config"])
    generator = SuppMaterialTablesGenerator(
        bins_stay_length=bins_stay_length,
        n_test_uniqueness=n_test_uniqueness,
        seed=random_state,
        tables=output_paths.keys(),
    )
    tables = generator(data=data)

    for table_name, output_path in output_paths.items():
        tables[table_name].to_pickle(output_path)
        logger.info(f"Results {table_name} saved at : {output_path}")


if __name__ == "__main__":
    app()
