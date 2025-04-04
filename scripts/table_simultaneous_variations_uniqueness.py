from typing import List

from confit import Cli

from privacy import registry

app = Cli(pretty_exceptions_show_locals=False)

from typing import Optional

from loguru import logger

from privacy.misc.utils import DataModule
from privacy.pipelines.table_simultaneous_variations_uniqueness import get_table_simultaneous_variations_uniqueness


@app.command(name="table_simultaneous_variations_uniqueness")
def main(
    output_path: str,
    list_shift_clinical: List[int],
    list_shift_birth: List[int],
    n: int = 50,
    batch_size: Optional[int] = 50,
    max_workers: int = 2,
    config_meta=None,
    **kwargs,
):
    data = DataModule(conf=config_meta["resolved_config"])

    table = get_table_simultaneous_variations_uniqueness(
        data=data,
        seed=config_meta["resolved_config"]["general"]["seed"],
        batch_size=batch_size,
        list_shift_clinical=list_shift_clinical,
        list_shift_birth=list_shift_birth,
        n=n,
        max_workers=max_workers,
    )

    table.to_csv(output_path)
    logger.info(f"Results saved at : {output_path}")


if __name__ == "__main__":
    app()
