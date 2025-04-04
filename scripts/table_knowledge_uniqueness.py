from confit import Cli

from privacy import registry

app = Cli(pretty_exceptions_show_locals=False)

from loguru import logger

from privacy.misc.utils import DataModule
from privacy.pipelines.table_knowledge_uniqueness import get_table_knowledge_uniqueness


@app.command(name="table_knowledge_uniqueness")
def main(
    output_path: str,
    batch_size: int = 40,
    config_meta=None,
):
    data = DataModule(conf=config_meta["resolved_config"])

    table = get_table_knowledge_uniqueness(
        data=data,
        seed=config_meta["resolved_config"]["general"]["seed"],
        batch_size=batch_size,
        output_path=output_path,
    )

    table.to_csv(output_path)
    logger.info(f"Results saved at : {output_path}")


if __name__ == "__main__":
    app()
